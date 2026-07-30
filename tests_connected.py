"""
Tests de la partition a territoires CONNEXES (ortools_ors_matrix_connected).

Lancement :
    python -m unittest tests_connected -v

Aucun acces reseau. flask, numpy, requests et sklearn sont bouchonnes avant
l'import d'app.py ; ortools est laisse absent, ce qui fait basculer
ORTOOLS_AVAILABLE a False. La geometrie, le graphe, la reparation et la
recherche locale n'en dependent pas et sont testes tels quels.
"""

import math
import statistics
import sys
import types
import unittest


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class _FakeFlask:
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):
        def deco(fn):
            return fn
        return deco

    def run(self, *a, **k):
        pass


_stub("flask", Flask=_FakeFlask, request=None, jsonify=lambda *a, **k: None)
_stub("requests", post=lambda *a, **k: None)
_np = _stub("numpy")
_np.mean = lambda seq: statistics.fmean(list(seq))
_np.array = lambda x: x
sys.modules["sklearn"] = types.ModuleType("sklearn")
_stub("sklearn.cluster", KMeans=object)
_stub("sklearn.metrics", silhouette_score=lambda *a, **k: 0.0)

import app  # noqa: E402


# ----------------------------------------------------------------- fixtures

def P(pid, lat, lon):
    return {"id": str(pid), "lat": lat, "lon": lon, "address": "pt " + str(pid)}


def euclid_matrix(points):
    xy = app._local_xy(points, list(range(len(points))))
    n = len(points)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i][j] = math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1])
    return m


def two_blobs(per_side=6, gap=0.03):
    """Deux amas nettement separes, depot en index 0."""
    pts = [P("DEP", 45.50, 4.40)]
    for k in range(per_side):
        pts.append(P("W%d" % k, 45.440 + 0.0015 * (k % 3), 4.380 + 0.0015 * (k // 3)))
    for k in range(per_side):
        pts.append(P("E%d" % k, 45.440 + 0.0015 * (k % 3), 4.380 + gap + 0.0015 * (k // 3)))
    return pts


def u_shape():
    """Forme en U : une droite coupe mal, une partition connexe existe.
    Branche gauche, fond, branche droite."""
    pts = [P("DEP", 45.50, 4.40)]
    for k in range(5):                       # branche gauche, du haut vers le bas
        pts.append(P("L%d" % k, 45.450 - 0.004 * k, 4.380))
    for k in range(4):                       # fond du U
        pts.append(P("B%d" % k, 45.430, 4.382 + 0.004 * k))
    for k in range(5):                       # branche droite, du bas vers le haut
        pts.append(P("R%d" % k, 45.434 + 0.004 * k, 4.398))
    return pts


def deliveries(points, start=0, end=0):
    d = {start, end}
    return [i for i in range(len(points)) if i not in d]


# ------------------------------------------------------------------- tests

class TestGraph(unittest.TestCase):

    def test_graph_is_globally_connected(self):
        """Le graphe global doit toujours etre connexe, quel que soit
        l'eloignement des amas. L'arbre couvrant n'ajoute rien lorsque les kNN
        suffisent deja : mst_edges peut legitimement valoir 0."""
        for gap in (0.01, 0.05, 0.20):
            pts = two_blobs(6, gap=gap)
            idx = deliveries(pts)
            adj, meta = app.build_geo_graph(pts, idx)
            self.assertTrue(meta["connected"],
                            "graphe non connexe pour gap=%s" % gap)
            self.assertTrue(app._graph_connected(idx, adj))
            self.assertGreaterEqual(meta["mst_edges"], 0)

    def test_graph_is_undirected_and_deterministic(self):
        pts = two_blobs(5)
        idx = deliveries(pts)
        a1, m1 = app.build_geo_graph(pts, idx)
        a2, m2 = app.build_geo_graph(pts, idx)
        self.assertEqual(m1, m2)
        self.assertEqual({k: sorted(v) for k, v in a1.items()},
                         {k: sorted(v) for k, v in a2.items()})
        for i, nbs in a1.items():
            for j in nbs:
                self.assertIn(i, a1[j], "graphe non oriente : arete manquante")

    def test_04_isolated_island_is_detected(self):
        adj = {1: {2}, 2: {1}, 3: set()}      # 3 est une ile
        info = app.is_connected_partition([1, 2, 3], adj)
        self.assertFalse(info["connected"])
        self.assertEqual(info["component_count"], 2)
        self.assertEqual(info["component_sizes"], [2, 1])

    def test_05_multiple_components(self):
        adj = {1: {2}, 2: {1}, 3: {4}, 4: {3}, 5: set()}
        info = app.is_connected_partition([1, 2, 3, 4, 5], adj)
        self.assertEqual(info["component_count"], 3)
        self.assertEqual(info["component_sizes"], [2, 2, 1])
        self.assertFalse(info["connected"])

    def test_single_component_is_connected(self):
        adj = {1: {2}, 2: {1, 3}, 3: {2}}
        info = app.is_connected_partition([1, 2, 3], adj)
        self.assertTrue(info["connected"])
        self.assertEqual(info["component_count"], 1)


class TestRepair(unittest.TestCase):

    def test_01_two_natural_groups(self):
        pts = two_blobs(6)
        idx = deliveries(pts)
        adj, _ = app.build_geo_graph(pts, idx)
        mat = euclid_matrix(pts)
        ga, gb = app._two_means_partition(pts, idx, len(idx) // 2)
        ga, gb, ok, _ = app.repair_to_connected(ga, gb, adj, pts, mat, 0, 0, len(idx) // 2)
        self.assertTrue(ok)
        self.assertTrue(app.is_connected_partition(ga, adj)["connected"])
        self.assertTrue(app.is_connected_partition(gb, adj)["connected"])

    def test_02_u_shape_has_connected_partition(self):
        pts = u_shape()
        idx = deliveries(pts)
        adj, _ = app.build_geo_graph(pts, idx)
        mat = euclid_matrix(pts)
        target = len(idx) // 2
        ga, gb = app._two_means_partition(pts, idx, target)
        ga, gb, ok, _ = app.repair_to_connected(ga, gb, adj, pts, mat, 0, 0, target)
        self.assertTrue(ok, "une partition connexe doit exister sur une forme en U")
        self.assertEqual(len(ga), target)
        self.assertTrue(app.is_connected_partition(ga, adj)["connected"])
        self.assertTrue(app.is_connected_partition(gb, adj)["connected"])

    def test_03_interleaved_partition_is_repaired(self):
        """Partition volontairement imbriquee, comme celle que produisait
        l'ancien ORS Matrix. Graphe en CHAINE explicite : un point sur deux
        rend chaque groupe totalement morcele, la reparation a donc du travail."""
        pts = [P("DEP", 45.50, 4.40)]
        for k in range(6):
            pts.append(P(k, 45.440 + 0.002 * k, 4.380))
        idx = deliveries(pts)                       # 1..6
        adj = {i: set() for i in idx}
        for a, b in zip(idx, idx[1:]):              # chaine 1-2-3-4-5-6
            adj[a].add(b)
            adj[b].add(a)
        mat = euclid_matrix(pts)
        target = 3

        ga, gb = idx[0::2], idx[1::2]               # [1,3,5] et [2,4,6]
        self.assertEqual(app.is_connected_partition(ga, adj)["component_count"], 3)
        self.assertEqual(app.is_connected_partition(gb, adj)["component_count"], 3)

        ga2, gb2, ok, moves = app.repair_to_connected(
            ga, gb, adj, pts, mat, 0, 0, target)
        self.assertTrue(ok, "la reparation doit aboutir")
        self.assertTrue(app.is_connected_partition(ga2, adj)["connected"])
        self.assertTrue(app.is_connected_partition(gb2, adj)["connected"])
        self.assertEqual(len(ga2), target)
        self.assertEqual(set(ga2) | set(gb2), set(idx))
        self.assertEqual(set(ga2) & set(gb2), set())
        self.assertGreater(moves, 0, "des deplacements etaient necessaires")

    def test_08_09_no_loss_no_duplicate_after_repair(self):
        pts = two_blobs(7)
        idx = deliveries(pts)
        adj, _ = app.build_geo_graph(pts, idx)
        mat = euclid_matrix(pts)
        target = len(idx) // 2
        ga, gb, ok, _ = app.repair_to_connected(
            idx[0::2], idx[1::2], adj, pts, mat, 0, 0, target)
        self.assertTrue(ok)
        self.assertEqual(set(ga) | set(gb), set(idx), "point perdu")
        self.assertEqual(set(ga) & set(gb), set(), "point duplique")

    def test_10_determinism(self):
        pts = two_blobs(6)
        idx = deliveries(pts)
        adj, _ = app.build_geo_graph(pts, idx)
        mat = euclid_matrix(pts)
        target = len(idx) // 2
        r1 = app.repair_to_connected(idx[0::2], idx[1::2], adj, pts, mat, 0, 0, target)
        r2 = app.repair_to_connected(idx[0::2], idx[1::2], adj, pts, mat, 0, 0, target)
        self.assertEqual(r1[0], r2[0])
        self.assertEqual(r1[1], r2[1])

        s1 = app.connected_local_search(r1[0], r1[1], adj, mat, 0, 0)
        s2 = app.connected_local_search(r2[0], r2[1], adj, mat, 0, 0)
        self.assertEqual(s1[0], s2[0])
        self.assertEqual(s1[2], s2[2])


class TestLocalSearch(unittest.TestCase):

    def test_local_search_preserves_cardinality_and_connectivity(self):
        pts = two_blobs(6)
        idx = deliveries(pts)
        adj, _ = app.build_geo_graph(pts, idx)
        mat = euclid_matrix(pts)
        target = len(idx) // 2
        ga, gb, ok, _ = app.repair_to_connected(
            idx[0::2], idx[1::2], adj, pts, mat, 0, 0, target)
        self.assertTrue(ok)
        na, nb, cost, swaps = app.connected_local_search(ga, gb, adj, mat, 0, 0)
        self.assertEqual(len(na), len(ga))
        self.assertEqual(len(nb), len(gb))
        self.assertEqual(set(na) | set(nb), set(idx))
        self.assertEqual(set(na) & set(nb), set())
        self.assertTrue(app.is_connected_partition(na, adj)["connected"])
        self.assertTrue(app.is_connected_partition(nb, adj)["connected"])

    def test_local_search_never_degrades(self):
        pts = two_blobs(6)
        idx = deliveries(pts)
        adj, _ = app.build_geo_graph(pts, idx)
        mat = euclid_matrix(pts)
        target = len(idx) // 2
        ga, gb, _, _ = app.repair_to_connected(
            idx[0::2], idx[1::2], adj, pts, mat, 0, 0, target)
        before = (app._estimate_group_cost(mat, ga, 0, 0, False)[0]
                  + app._estimate_group_cost(mat, gb, 0, 0, False)[0])
        _, _, after, _ = app.connected_local_search(ga, gb, adj, mat, 0, 0)
        self.assertLessEqual(after, before + 1e-9)

    def test_boundary_metrics(self):
        adj = {1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
        pts = [P("DEP", 45.5, 4.4), P(1, 45.44, 4.38), P(2, 45.441, 4.381),
               P(3, 45.442, 4.382), P(4, 45.443, 4.383)]
        b = app.boundary_metrics([1, 2], [3, 4], adj, pts)
        self.assertEqual(b["cut_edges"], 1)       # seule l'arete 2-3 traverse
        self.assertGreater(b["cut_length_m"], 0.0)


class TestCardinality(unittest.TestCase):

    def _run(self, n_points):
        pts = [P("DEP", 45.4666, 4.3903)]
        for k in range(n_points):
            pts.append(P(k, 45.400 + 0.0018 * (k % 10), 4.370 + 0.0022 * (k // 10)))
        idx = deliveries(pts)
        adj, _ = app.build_geo_graph(pts, idx)
        mat = euclid_matrix(pts)
        target = len(idx) // 2
        ga, gb, ok, _ = app.repair_to_connected(
            idx[0::2], idx[1::2], adj, pts, mat, 0, 0, target)
        return idx, adj, ga, gb, ok, target

    def test_06_sixty_points_is_30_30(self):
        idx, adj, ga, gb, ok, target = self._run(60)
        self.assertTrue(ok)
        self.assertEqual(target, 30)
        self.assertEqual([len(ga), len(gb)], [30, 30])
        self.assertEqual(set(ga) | set(gb), set(idx))
        self.assertEqual(set(ga) & set(gb), set())
        self.assertTrue(app.is_connected_partition(ga, adj)["connected"])
        self.assertTrue(app.is_connected_partition(gb, adj)["connected"])

    def test_07_fifty_eight_points_is_29_29(self):
        idx, adj, ga, gb, ok, target = self._run(58)
        self.assertTrue(ok)
        self.assertEqual([len(ga), len(gb)], [29, 29])

    def test_odd_count_differs_by_one(self):
        idx, adj, ga, gb, ok, target = self._run(59)
        self.assertTrue(ok)
        self.assertEqual(abs(len(ga) - len(gb)), 1)
        self.assertEqual(len(ga) + len(gb), 59)


class TestWiringAndInvariants(unittest.TestCase):

    def setUp(self):
        with open("app.py", encoding="utf-8") as fh:
            self.src = fh.read()

    def test_strategy_registered(self):
        self.assertIn("ortools_ors_matrix_connected", app.VALID_STRATEGIES)
        self.assertIn('partition_solver = "connected_graph_partition"', self.src)

    def test_11_no_matrix_call_during_local_generation(self):
        """Aucune generation ni modification de partition ne doit declencher
        un appel Matrix : la matrice deja chargee est reutilisee."""
        start = self.src.index("# 4e. PARTITION CONNEXE")
        end = self.src.index("def ortools_partition_ors_matrix_connected")
        block = self.src[start:end]
        for forbidden in ("_post_matrix", "_build_full_matrix_chunked",
                          "_fetch_ors_matrix", "requests."):
            self.assertNotIn(forbidden, block,
                             "appel reseau dans la generation locale : " + forbidden)

    def test_12_vroom_failure_keeps_ortools(self):
        body = self.src[self.src.index("def ortools_partition_ors_matrix_connected"):
                        self.src.index("def _tsp_order_ortools")]
        self.assertIn("connected_fallback_used", body)
        self.assertIn("kept OR-Tools order", body)
        # l'echec Vroom sort de la boucle sans lever ni annuler l'incumbent
        self.assertIn("break", body[body.index("if ra is None or rb is None:"):])

    def test_13_swaps_locked_for_connected(self):
        self.assertIn('swap_lock_reason = "connected_partition_locked"', self.src)
        self.assertIn("connected_partition_locked", app.SWAP_STOP_REASONS)
        locked = self.src.index("if routes_idx and vroom_ok and membership_locked:")
        swaps = self.src.index("elif routes_idx and vroom_ok:")
        self.assertLess(locked, swaps)
        self.assertNotIn("post_process_swaps(", self.src[locked:swaps])

    def test_14_kmeans_unchanged(self):
        for name in ("kmeans_partition", "_balance_groups", "_create_sub_clusters"):
            self.assertIn("def " + name + "(", self.src)

    def test_15_haversine_unchanged(self):
        body = self.src[self.src.index("def ortools_partition_haversine"):
                        self.src.index("# 4c. MATRICE ORS COMPLETE")]
        self.assertIn("_solve_cvrp_ortools(", body)
        self.assertNotIn("connected", body)

    def test_16_sweep_mode_unchanged(self):
        self.assertIn("def enumerate_territorial_partitions(", self.src)
        self.assertIn('"territorial_method": "sweep_line_projection"', self.src)
        body = self.src[self.src.index("def ortools_partition_ors_matrix("):
                        self.src.index("# 4e. PARTITION CONNEXE")]
        self.assertNotIn("build_geo_graph", body,
                         "le mode sweep ne doit pas dependre du graphe")

    def test_17_d1_d2_d3_intact(self):
        self.assertIn("ORTOOLS_SOLUTION_LIMIT = 75", self.src)
        self.assertIn("ORTOOLS_TIME_LIMIT_S = 25", self.src)
        self.assertIn("api.heigit.org/vroom/v0", self.src)
        self.assertIn("key = tuple(pts)", self.src)        # cache exact D-3
        self.assertIn("duration_s", self.src)              # garde-fou D-2
        self.assertIn("SWAP_MAX_CANDIDATES = 50", self.src)

    def test_no_balancing_in_selection(self):
        body = self.src[self.src.index("def _selection_key"):
                        self.src.index("def ortools_partition_ors_matrix_connected")]
        self.assertNotIn("abs(", body,
                         "aucun terme d'equilibrage dans la cle de selection")


if __name__ == "__main__":
    unittest.main(verbosity=2)
