import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantLock;

public class Main {
    static final int DIMS = 16;

    // =====================================================================
    //  DATA TYPES
    // =====================================================================
    static class VectorItem {
        int id;
        String metadata;
        String category;
        float[] emb;
        VectorItem(int id, String metadata, String category, float[] emb) {
            this.id = id; this.metadata = metadata; this.category = category; this.emb = emb;
        }
    }

    static class Hit implements Comparable<Hit> {
        float distance;
        int id;
        Hit(float distance, int id) { this.distance = distance; this.id = id; }
        @Override public int compareTo(Hit o) { return Float.compare(this.distance, o.distance); }
    }

    // =====================================================================
    //  DISTANCE METRICS
    // =====================================================================
    interface DistFn { float compute(float[] a, float[] b); }

    static final DistFn EUCLIDEAN = (a, b) -> {
        float s = 0;
        for (int i = 0; i < a.length; i++) { float d = a[i] - b[i]; s += d * d; }
        return (float) Math.sqrt(s);
    };

    static final DistFn COSINE = (a, b) -> {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - (float) (dot / (Math.sqrt(na) * Math.sqrt(nb)));
    };

    static final DistFn MANHATTAN = (a, b) -> {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    };

    static DistFn getDistFn(String m) {
        if ("cosine".equals(m)) return COSINE;
        if ("manhattan".equals(m)) return MANHATTAN;
        return EUCLIDEAN;
    }

    // =====================================================================
    //  BRUTE FORCE
    // =====================================================================
    static class BruteForce {
        List<VectorItem> items = new ArrayList<>();
        void insert(VectorItem v) { items.add(v); }
        void remove(int id) { items.removeIf(v -> v.id == id); }
        List<Hit> knn(float[] q, int k, DistFn dist) {
            PriorityQueue<Hit> pq = new PriorityQueue<>(Collections.reverseOrder());
            for (VectorItem v : items) {
                pq.offer(new Hit(dist.compute(q, v.emb), v.id));
                if (pq.size() > k) pq.poll();
            }
            List<Hit> res = new ArrayList<>();
            while (!pq.isEmpty()) res.add(pq.poll());
            Collections.reverse(res);
            return res;
        }
    }

    // =====================================================================
    //  KD-TREE
    // =====================================================================
    static class KDNode {
        VectorItem item; KDNode left, right;
        KDNode(VectorItem item) { this.item = item; }
    }

    static class KDTree {
        KDNode root; int dims;
        KDTree(int dims) { this.dims = dims; }
        void insert(VectorItem v) { root = ins(root, v, 0); }
        private KDNode ins(KDNode n, VectorItem v, int d) {
            if (n == null) return new KDNode(v);
            int ax = d % dims;
            if (v.emb[ax] < n.item.emb[ax]) n.left = ins(n.left, v, d + 1);
            else n.right = ins(n.right, v, d + 1);
            return n;
        }
        void rebuild(List<VectorItem> items) {
            root = null;
            for (VectorItem v : items) insert(v);
        }
        List<Hit> knn(float[] q, int k, DistFn dist) {
            PriorityQueue<Hit> heap = new PriorityQueue<>(Collections.reverseOrder());
            knn(root, q, k, 0, dist, heap);
            List<Hit> res = new ArrayList<>();
            while (!heap.isEmpty()) res.add(heap.poll());
            Collections.reverse(res);
            return res;
        }
        private void knn(KDNode n, float[] q, int k, int d, DistFn dist, PriorityQueue<Hit> heap) {
            if (n == null) return;
            float dn = dist.compute(q, n.item.emb);
            if (heap.size() < k || dn < heap.peek().distance) {
                heap.offer(new Hit(dn, n.item.id));
                if (heap.size() > k) heap.poll();
            }
            int ax = d % dims;
            float diff = q[ax] - n.item.emb[ax];
            KDNode closer = diff < 0 ? n.left : n.right;
            KDNode farther = diff < 0 ? n.right : n.left;
            knn(closer, q, k, d + 1, dist, heap);
            if (heap.size() < k || Math.abs(diff) < heap.peek().distance)
                knn(farther, q, k, d + 1, dist, heap);
        }
    }

    // =====================================================================
    //  HNSW (Hierarchical Navigable Small World)
    // =====================================================================
    static class HNSW {
        static class Node {
            VectorItem item; int maxLyr; List<List<Integer>> nbrs;
            Node(VectorItem item, int maxLyr) {
                this.item = item; this.maxLyr = maxLyr; this.nbrs = new ArrayList<>();
                for (int i = 0; i <= maxLyr; i++) nbrs.add(new ArrayList<>());
            }
        }
        Map<Integer, Node> G = new ConcurrentHashMap<>();
        int M, M0, efBuild; double mL; int topLayer = -1, entryPt = -1;
        Random rng = new Random(42);

        HNSW(int m, int efBuild) {
            this.M = m; this.M0 = 2 * m; this.efBuild = efBuild;
            this.mL = 1.0 / Math.log(m);
        }

        int randLevel() { return (int) Math.floor(-Math.log(rng.nextDouble()) * mL); }

        void insert(VectorItem item, DistFn dist) {
            int id = item.id, lvl = randLevel();
            Node node = new Node(item, lvl);
            G.put(id, node);
            if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

            int ep = entryPt;
            for (int lc = topLayer; lc > lvl; lc--) {
                List<Hit> W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = W.get(0).id;
            }
            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<Hit> W = searchLayer(item.emb, ep, efBuild, lc, dist);
                int maxM = (lc == 0) ? M0 : M;
                List<Integer> sel = new ArrayList<>();
                for (int i = 0; i < Math.min(W.size(), maxM); i++) sel.add(W.get(i).id);
                node.nbrs.get(lc).addAll(sel);

                for (int nid : sel) {
                    Node neighbor = G.get(nid);
                    if (neighbor == null) continue;
                    List<Integer> conn = neighbor.nbrs.get(lc);
                    conn.add(id);
                    if (conn.size() > maxM) {
                        List<Hit> ds = new ArrayList<>();
                        for (int c : conn) {
                            Node cNode = G.get(c);
                            if (cNode != null) ds.add(new Hit(dist.compute(neighbor.item.emb, cNode.item.emb), c));
                        }
                        Collections.sort(ds);
                        conn.clear();
                        for (int i = 0; i < maxM && i < ds.size(); i++) conn.add(ds.get(i).id);
                    }
                }
                if (!W.isEmpty()) ep = W.get(0).id;
            }
            if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
        }

        List<Hit> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
            Set<Integer> vis = new HashSet<>();
            PriorityQueue<Hit> cands = new PriorityQueue<>();
            PriorityQueue<Hit> found = new PriorityQueue<>(Collections.reverseOrder());

            float d0 = dist.compute(q, G.get(ep).item.emb);
            vis.add(ep); cands.offer(new Hit(d0, ep)); found.offer(new Hit(d0, ep));

            while (!cands.isEmpty()) {
                Hit c = cands.poll();
                if (found.size() >= ef && c.distance > found.peek().distance) break;
                Node cNode = G.get(c.id);
                if (lyr >= cNode.nbrs.size()) continue;

                for (int nid : cNode.nbrs.get(lyr)) {
                    if (!vis.add(nid) || !G.containsKey(nid)) continue;
                    float nd = dist.compute(q, G.get(nid).item.emb);
                    if (found.size() < ef || nd < found.peek().distance) {
                        cands.offer(new Hit(nd, nid)); found.offer(new Hit(nd, nid));
                        if (found.size() > ef) found.poll();
                    }
                }
            }
            List<Hit> res = new ArrayList<>();
            while (!found.isEmpty()) res.add(found.poll());
            Collections.reverse(res);
            return res;
        }

        List<Hit> knn(float[] q, int k, int ef, DistFn dist) {
            if (entryPt == -1) return new ArrayList<>();
            int ep = entryPt;
            for (int lc = topLayer; lc > 0; lc--) {
                List<Hit> W = searchLayer(q, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = W.get(0).id;
            }
            List<Hit> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
            return W.size() > k ? W.subList(0, k) : W;
        }

        void remove(int id) {
            if (!G.containsKey(id)) return;
            for (Node nd : G.values()) {
                for (List<Integer> layer : nd.nbrs) layer.remove(Integer.valueOf(id));
            }
            if (entryPt == id) {
                entryPt = -1;
                for (int nid : G.keySet()) if (nid != id) { entryPt = nid; break; }
            }
            G.remove(id);
        }
    }

    // =====================================================================
    //  VECTOR DATABASE (Demo Vectors)
    // =====================================================================
    static class VectorDB {
        Map<Integer, VectorItem> store = new HashMap<>();
        BruteForce bf = new BruteForce();
        KDTree kdt = new KDTree(DIMS);
        HNSW hnsw = new HNSW(16, 200);
        ReentrantLock mu = new ReentrantLock();
        int nextId = 1;

        int insert(String meta, String cat, float[] emb, DistFn dist) {
            mu.lock();
            try {
                VectorItem v = new VectorItem(nextId++, meta, cat, emb);
                store.put(v.id, v);
                bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
                return v.id;
            } finally { mu.unlock(); }
        }

        boolean remove(int id) {
            mu.lock();
            try {
                if (!store.containsKey(id)) return false;
                store.remove(id); bf.remove(id); hnsw.remove(id);
                kdt.rebuild(new ArrayList<>(store.values()));
                return true;
            } finally { mu.unlock(); }
        }

        static class SearchOut { List<Hit> hits; long us; String algo, metric; }

        SearchOut search(float[] q, int k, String metric, String algo) {
            mu.lock();
            try {
                DistFn dfn = getDistFn(metric);
                long t0 = System.nanoTime();
                List<Hit> raw;
                if ("bruteforce".equals(algo)) raw = bf.knn(q, k, dfn);
                else if ("kdtree".equals(algo)) raw = kdt.knn(q, k, dfn);
                else raw = hnsw.knn(q, k, 50, dfn);
                long us = (System.nanoTime() - t0) / 1000;
                SearchOut out = new SearchOut();
                out.hits = raw; out.us = us; out.algo = algo; out.metric = metric;
                return out;
            } finally { mu.unlock(); }
        }

        static class BenchOut { long bfUs, kdUs, hnswUs; int n; }

        BenchOut benchmark(float[] q, int k, String metric) {
            mu.lock();
            try {
                DistFn dfn = getDistFn(metric);
                BenchOut b = new BenchOut();
                long t0 = System.nanoTime(); bf.knn(q, k, dfn); b.bfUs = (System.nanoTime() - t0) / 1000;
                t0 = System.nanoTime(); kdt.knn(q, k, dfn); b.kdUs = (System.nanoTime() - t0) / 1000;
                t0 = System.nanoTime(); hnsw.knn(q, k, 50, dfn); b.hnswUs = (System.nanoTime() - t0) / 1000;
                b.n = store.size();
                return b;
            } finally { mu.unlock(); }
        }
    }

    // =====================================================================
    //  OLLAMA CLIENT (REST Integration)
    // =====================================================================
    static class OllamaClient {
        HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
        String host = "http://127.0.0.1:11434";
        String embedModel = "nomic-embed-text";
        String genModel = "llama3.2";

        String esc(String s) {
            return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
        }

        boolean isAvailable() {
            try {
                HttpRequest req = HttpRequest.newBuilder().uri(URI.create(host + "/api/tags")).GET().build();
                return client.send(req, HttpResponse.BodyHandlers.discarding()).statusCode() == 200;
            } catch (Exception e) { return false; }
        }

        float[] embed(String text) {
            try {
                String body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
                HttpRequest req = HttpRequest.newBuilder().uri(URI.create(host + "/api/embeddings"))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return null;
                String b = res.body();
                int s = b.indexOf("\"embedding\":[") + 13, e = b.indexOf("]", s);
                if (s < 13 || e == -1) return null;
                String[] parts = b.substring(s, e).split(",");
                float[] emb = new float[parts.length];
                for (int i = 0; i < parts.length; i++) emb[i] = Float.parseFloat(parts[i].trim());
                return emb;
            } catch (Exception e) { return null; }
        }

        String generate(String prompt) {
            try {
                String body = "{\"model\":\"" + genModel + "\",\"prompt\":\"" + esc(prompt) + "\",\"stream\":false}";
                HttpRequest req = HttpRequest.newBuilder().uri(URI.create(host + "/api/generate"))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return "ERROR: Ollama unavailable.";
                return extractStr(res.body(), "response");
            } catch (Exception e) { return "ERROR: Ollama unavailable."; }
        }
    }

    // =====================================================================
    //  DOCUMENT DATABASE (RAG)
    // =====================================================================
    static class DocItem {
        int id; String title, text; float[] emb;
        DocItem(int id, String title, String text, float[] emb) {
            this.id = id; this.title = title; this.text = text; this.emb = emb;
        }
    }

    static class DocumentDB {
        Map<Integer, DocItem> store = new HashMap<>();
        HNSW hnsw = new HNSW(16, 200);
        BruteForce bf = new BruteForce();
        ReentrantLock mu = new ReentrantLock();
        int nextId = 1, dims = 0;

        int insert(String title, String text, float[] emb) {
            mu.lock();
            try {
                if (dims == 0) dims = emb.length;
                DocItem item = new DocItem(nextId++, title, text, emb);
                store.put(item.id, item);
                VectorItem vi = new VectorItem(item.id, title, "doc", emb);
                hnsw.insert(vi, COSINE); bf.insert(vi);
                return item.id;
            } finally { mu.unlock(); }
        }

        boolean remove(int id) {
            mu.lock();
            try {
                if (store.remove(id) == null) return false;
                hnsw.remove(id); bf.remove(id);
                return true;
            } finally { mu.unlock(); }
        }

        List<Hit> search(float[] q, int k) {
            mu.lock();
            try {
                if (store.isEmpty()) return new ArrayList<>();
                List<Hit> raw = store.size() < 10 ? bf.knn(q, k, COSINE) : hnsw.knn(q, k, 50, COSINE);
                List<Hit> res = new ArrayList<>();
                for (Hit h : raw) if (store.containsKey(h.id) && h.distance <= 0.7f) res.add(h);
                return res;
            } finally { mu.unlock(); }
        }
    }

    // =====================================================================
    //  JSON HELPERS & UTILS
    // =====================================================================
    static String jS(String s) {
        if (s == null) return "\"\"";
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + "\"";
    }

    static String jVec(float[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(","); sb.append(String.format(Locale.US, "%.4f", v[i]));
        }
        return sb.append("]").toString();
    }

    static String extractStr(String json, String key) {
        String target = "\"" + key + "\"";
        int idx = json.indexOf(target);
        if (idx == -1) return "";
        idx = json.indexOf(":", idx);
        int start = json.indexOf("\"", idx) + 1;
        int end = start;
        while (end < json.length()) {
            if (json.charAt(end) == '"' && json.charAt(end - 1) != '\\') break;
            end++;
        }
        return json.substring(start, end).replace("\\\"", "\"").replace("\\n", "\n").replace("\\\\", "\\");
    }

    static int extractInt(String json, String key, int def) {
        String target = "\"" + key + "\"";
        int idx = json.indexOf(target);
        if (idx == -1) return def;
        idx = json.indexOf(":", idx) + 1;
        int end = idx;
        while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == ' ')) end++;
        try { return Integer.parseInt(json.substring(idx, end).trim()); } catch (Exception e) { return def; }
    }

    static float[] extractArr(String json, String key) {
        String target = "\"" + key + "\"";
        int idx = json.indexOf(target);
        if (idx == -1) return null;
        int s = json.indexOf("[", idx) + 1, e = json.indexOf("]", s);
        if (s == 0 || e == -1) return null;
        String[] parts = json.substring(s, e).split(",");
        float[] arr = new float[parts.length];
        for (int i = 0; i < parts.length; i++) arr[i] = Float.parseFloat(parts[i].trim());
        return arr;
    }

    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.split("\\s+");
        List<String> chunks = new ArrayList<>();
        if (words.length <= chunkWords) { chunks.add(text); return chunks; }
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) { if (j > i) sb.append(" "); sb.append(words[j]); }
            chunks.add(sb.toString());
            if (end == words.length) break;
        }
        return chunks;
    }

    static void sendJson(HttpExchange ex, String json) throws IOException {
        ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().add("Content-Type", "application/json");
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(200, bytes.length);
        ex.getResponseBody().write(bytes);
        ex.getResponseBody().close();
    }

    static Map<String, String> parseQuery(String query) {
        Map<String, String> res = new HashMap<>();
        if (query == null) return res;
        for (String param : query.split("&")) {
            String[] p = param.split("=");
            if (p.length > 1) res.put(p[0], p[1]);
        }
        return res;
    }

    static String readBody(HttpExchange ex) throws IOException {
        return new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    // =====================================================================
    //  MAIN HTTP SERVER
    // =====================================================================
    public static void main(String[] args) throws IOException {
        VectorDB db = new VectorDB();
        DocumentDB docDB = new DocumentDB();
        OllamaClient ollama = new OllamaClient();

        // Load 20 Demo Vectors
        float[][] demos = {
            {0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f},
            {0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f},
            {0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f},
            {0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f},
            {0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f},
            {0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f},
            {0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f},
            {0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f},
            {0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f},
            {0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f},
            {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f},
            {0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f},
            {0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f},
            {0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f},
            {0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f},
            {0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f},
            {0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f},
            {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f},
            {0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f},
            {0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}
        };
        String[] metas = {"Linked List","BST","Dynamic Programming","Graph BFS","Hash Table",
                "Calculus","Linear Algebra","Probability","Number Theory","Combinatorics",
                "Pizza","Sushi","Ramen","Tacos","Croissant",
                "Basketball","Football","Tennis","Chess","Swimming"};
        String[] cats = {"cs","cs","cs","cs","cs","math","math","math","math","math","food","food","food","food","food","sports","sports","sports","sports","sports"};
        for (int i = 0; i < 20; i++) db.insert(metas[i], cats[i], demos[i], COSINE);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:8080");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE"));

        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);
        server.setExecutor(Executors.newCachedThreadPool());

        // Preflight CORS handler for all methods
        server.createContext("/", ex -> {
            if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
                ex.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
                ex.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
                ex.sendResponseHeaders(204, -1);
                return;
            }
            if ("/".equals(ex.getRequestURI().getPath())) {
                File file = new File("index.html");
                if (!file.exists()) { ex.sendResponseHeaders(404, -1); return; }
                ex.getResponseHeaders().add("Content-Type", "text/html");
                byte[] bytes = java.nio.file.Files.readAllBytes(file.toPath());
                ex.sendResponseHeaders(200, bytes.length);
                ex.getResponseBody().write(bytes);
                ex.getResponseBody().close();
            }
        });

        // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────
        server.createContext("/items", ex -> {
            StringBuilder sb = new StringBuilder("[");
            int count = 0;
            for (VectorItem v : db.store.values()) {
                if (count++ > 0) sb.append(",");
                sb.append("{\"id\":").append(v.id).append(",\"metadata\":").append(jS(v.metadata))
                  .append(",\"category\":").append(jS(v.category)).append(",\"embedding\":").append(jVec(v.emb)).append("}");
            }
            sendJson(ex, sb.append("]").toString());
        });

        server.createContext("/search", ex -> {
            Map<String, String> q = parseQuery(ex.getRequestURI().getQuery());
            String[] vStrs = q.get("v").split(",");
            float[] vec = new float[DIMS];
            for (int i = 0; i < DIMS; i++) vec[i] = Float.parseFloat(vStrs[i]);
            int k = Integer.parseInt(q.getOrDefault("k", "5"));
            String algo = q.getOrDefault("algo", "hnsw"), metric = q.getOrDefault("metric", "cosine");

            VectorDB.SearchOut out = db.search(vec, k, metric, algo);
            StringBuilder sb = new StringBuilder("{\"results\":[");
            for (int i = 0; i < out.hits.size(); i++) {
                if (i > 0) sb.append(",");
                Hit h = out.hits.get(i);
                VectorItem vi = db.store.get(h.id);
                sb.append("{\"id\":").append(h.id).append(",\"metadata\":").append(jS(vi.metadata))
                  .append(",\"category\":").append(jS(vi.category)).append(",\"distance\":").append(h.distance)
                  .append(",\"embedding\":").append(jVec(vi.emb)).append("}");
            }
            sb.append("],\"latencyUs\":").append(out.us).append(",\"algo\":").append(jS(algo)).append(",\"metric\":").append(jS(metric)).append("}");
            sendJson(ex, sb.toString());
        });

        server.createContext("/insert", ex -> {
            String body = readBody(ex);
            String meta = extractStr(body, "metadata"), cat = extractStr(body, "category");
            float[] emb = extractArr(body, "embedding");
            int id = db.insert(meta, cat, emb, COSINE);
            sendJson(ex, "{\"id\":" + id + "}");
        });

        server.createContext("/delete/", ex -> {
            int id = Integer.parseInt(ex.getRequestURI().getPath().replace("/delete/", ""));
            sendJson(ex, "{\"ok\":" + db.remove(id) + "}");
        });

        server.createContext("/benchmark", ex -> {
            Map<String, String> q = parseQuery(ex.getRequestURI().getQuery());
            String[] vStrs = q.get("v").split(",");
            float[] vec = new float[DIMS];
            for (int i = 0; i < DIMS; i++) vec[i] = Float.parseFloat(vStrs[i]);
            VectorDB.BenchOut b = db.benchmark(vec, 5, q.getOrDefault("metric", "cosine"));
            sendJson(ex, "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs + ",\"hnswUs\":" + b.hnswUs + ",\"itemCount\":" + b.n + "}");
        });

        server.createContext("/hnsw-info", ex -> {
            int[] nodesPerLayer = new int[db.hnsw.topLayer + 2];
            int[] edgesPerLayer = new int[db.hnsw.topLayer + 2];
            for (HNSW.Node nd : db.hnsw.G.values()) {
                for (int l = 0; l <= nd.maxLyr; l++) {
                    nodesPerLayer[l]++;
                    if (l < nd.nbrs.size()) edgesPerLayer[l] += nd.nbrs.get(l).size();
                }
            }
            StringBuilder sb = new StringBuilder("{\"topLayer\":").append(db.hnsw.topLayer)
                    .append(",\"nodeCount\":").append(db.hnsw.G.size()).append(",\"nodesPerLayer\":[");
            for (int i=0; i<nodesPerLayer.length; i++) sb.append(i>0?",":"").append(nodesPerLayer[i]);
            sb.append("],\"edgesPerLayer\":[");
            for (int i=0; i<edgesPerLayer.length; i++) sb.append(i>0?",":"").append(edgesPerLayer[i]/2);
            sendJson(ex, sb.append("]}").toString());
        });

        // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────
        server.createContext("/doc/insert", ex -> {
            String body = readBody(ex);
            String title = extractStr(body, "title"), text = extractStr(body, "text");
            List<String> chunks = chunkText(text, 250, 30);
            List<Integer> ids = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                float[] emb = ollama.embed(chunks.get(i));
                if (emb == null) { sendJson(ex, "{\"error\":\"Ollama unavailable.\"}"); return; }
                String cTitle = chunks.size() > 1 ? title + " [" + (i+1) + "/" + chunks.size() + "]" : title;
                ids.add(docDB.insert(cTitle, chunks.get(i), emb));
            }
            sendJson(ex, "{\"chunks\":" + chunks.size() + ",\"dims\":" + docDB.dims + "}");
        });

        server.createContext("/doc/list", ex -> {
            StringBuilder sb = new StringBuilder("[");
            int c = 0;
            for (DocItem d : docDB.store.values()) {
                if (c++ > 0) sb.append(",");
                String prev = d.text.length() > 120 ? d.text.substring(0, 120) + "..." : d.text;
                sb.append("{\"id\":").append(d.id).append(",\"title\":").append(jS(d.title))
                  .append(",\"preview\":").append(jS(prev)).append(",\"words\":").append(d.text.split(" ").length).append("}");
            }
            sendJson(ex, sb.append("]").toString());
        });

        server.createContext("/doc/delete/", ex -> {
            int id = Integer.parseInt(ex.getRequestURI().getPath().replace("/doc/delete/", ""));
            sendJson(ex, "{\"ok\":" + docDB.remove(id) + "}");
        });

        server.createContext("/doc/search", ex -> {
            String body = readBody(ex);
            String q = extractStr(body, "question");
            int k = extractInt(body, "k", 3);
            float[] qEmb = ollama.embed(q);
            if (qEmb == null) { sendJson(ex, "{\"error\":\"Ollama unavailable\"}"); return; }
            List<Hit> hits = docDB.search(qEmb, k);
            StringBuilder sb = new StringBuilder("{\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) sb.append(",");
                DocItem d = docDB.store.get(hits.get(i).id);
                sb.append("{\"id\":").append(d.id).append(",\"title\":").append(jS(d.title)).append(",\"distance\":").append(hits.get(i).distance).append("}");
            }
            sendJson(ex, sb.append("]}").toString());
        });

        server.createContext("/doc/ask", ex -> {
            String body = readBody(ex);
            String q = extractStr(body, "question");
            int k = extractInt(body, "k", 3);
            float[] qEmb = ollama.embed(q);
            if (qEmb == null) { sendJson(ex, "{\"error\":\"Ollama unavailable\"}"); return; }
            List<Hit> hits = docDB.search(qEmb, k);
            StringBuilder ctx = new StringBuilder();
            StringBuilder ctxJson = new StringBuilder("[");
            for (int i = 0; i < hits.size(); i++) {
                DocItem d = docDB.store.get(hits.get(i).id);
                ctx.append("[").append(i+1).append("] ").append(d.title).append(":\n").append(d.text).append("\n\n");
                if (i > 0) ctxJson.append(",");
                ctxJson.append("{\"id\":").append(d.id).append(",\"title\":").append(jS(d.title))
                       .append(",\"text\":").append(jS(d.text)).append(",\"distance\":").append(hits.get(i).distance).append("}");
            }
            ctxJson.append("]");
            String prompt = "You are a helpful assistant. Answer the user's question directly. Use the provided context if it contains relevant information. If it doesn't, just use your own general knowledge. IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. Just answer the question naturally.\n\nContext:\n" 
                            + ctx.toString() + "Question: " + q + "\n\nAnswer:";
            String ans = ollama.generate(prompt);
            sendJson(ex, "{\"answer\":" + jS(ans) + ",\"model\":\"" + ollama.genModel + "\",\"contexts\":" + ctxJson.toString() + "}");
        });

        server.createContext("/status", ex -> {
            boolean up = ollama.isAvailable();
            sendJson(ex, "{\"ollamaAvailable\":" + up + ",\"embedModel\":\"" + ollama.embedModel + "\",\"genModel\":\"" + ollama.genModel + "\",\"docCount\":" + docDB.store.size() + ",\"docDims\":" + docDB.dims + "}");
        });

        server.start();
    }
}