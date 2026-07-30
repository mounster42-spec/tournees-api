"""
Tests COMPORTEMENTAUX de la diversification des partitions connexes.

Lancement :
    python -m unittest tests_connected_diversity -v

La generation de candidates est purement locale : geometrie, graphe de
voisinage et matrice ORS DEJA CHARGEE. Ces tests le verifient en faisant
exploser toute tentative d'appel reseau, puis controlent les invariants de
chaque source, la deduplication canonique, les budgets et le determinisme.
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


# --------------------------------------------------------------- fixtures

def P(pid, lat, lon):
    return {"id": str(pid), "lat": lat, "lon": lon, "address": "pt " + str(pid)}


def grid_points(n, cols=10, dlat=0.0018, dlon=0.0022):
    """Semis regulier reproductible, depot en index 0."""
    pts = [P("DEP", 45.4666, 4.3903)]
    for k in range(n):
        pts.append(P(k, 45.400 + dlat * (k % cols), 4.370 + dlon * (k // cols)))
    return pts


def scattered_points(n, seed=7):
    """Semis pseudo-aleatoire mais deterministe : pas de module random, donc
    pas de dependance a l'etat global de l'interpreteur."""
    pts = [P("DEP", 45.4666, 4.3903)]
    x = seed
    for k in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        a = x / 2147483648.0
        x = (1103515245 * x + 12345) % 2147483648
        b = x / 2147483648.0
        pts.append(P(k, 45.400 + 0.09 * a, 4.330 + 0.11 * b))
    return pts


def euclid_matrix(points, scale=1.0):
    xy = app._local_xy(points, list(range(len(points))))
    n = len(points)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i][j] = math.hypot(xy[i][0] - xy[j][0],
                                     xy[i][1] - xy[j][1]) * scale
    return m


def deliveries(points, start=0, end=0):
    d = {start, end}
    return [i for i in range(len(points)) if i not in d]


def chain_graph(nodes):
    """Graphe en chaine 1-2-3-... : les composantes y sont evidentes."""
    adj = {i: set() for i in nodes}
    for a, b in zip(nodes, nodes[1:]):
        adj[a].add(b)
        adj[b].add(a)
    return adj


def build(points, ors=True):
    """Graphe + matrices, comme en production mais sans reseau."""
    idx = deliveries(points)
    dur = euclid_matrix(points)
    dist = euclid_matrix(points, scale=1000.0)
    adj, gmeta = app.build_geo_graph(points, idx,
                                     dur_matrix=dur if ors else None)
    return idx, adj, gmeta, dur, dist


class _NoNetwork:
    """Fait echouer bruyamment tout appel reseau pendant la generation."""

    NAMES = ("_post_matrix", "_post_vroom", "_build_full_matrix_chunked",
             "_fetch_ors_matrix", "_resequence_single", "_call_vroom_multi")

    def __enter__(self):
        self.saved = {n: getattr(app, n) for n in self.NAMES}
        self.saved["requests.post"] = app.requests.post

        def boom(*a, **k):
            raise AssertionError("appel reseau pendant la generation locale")

        for n in self.NAMES:
            setattr(app, n, boom)
        app.requests.post = boom
        return self

    def __exit__(self, *exc):
        for n in self.NAMES:
            setattr(app, n, self.saved[n])
        app.requests.post = self.saved["requests.post"]
        return False


# ------------------------------------------------- 1-2 : cle canonique

class TestCanonicalKey(unittest.TestCase):

    def test_01_a_partition_and_its_mirror_share_one_key(self):
        a, b = [1, 2, 3], [4, 5, 6]
        self.assertEqual(app.canonical_partition_key(a, b),
                         app.canonical_partition_key(b, a))

    def test_02_two_visit_orders_are_one_partition(self):
        """La cle ne decrit QUE l'appartenance : deux tournees differentes sur
        les memes groupes ne comptent pas pour deux partitions."""
        k1 = app.canonical_partition_key([3, 1, 2], [6, 4, 5])
        k2 = app.canonical_partition_key([2, 3, 1], [5, 6, 4])
        self.assertEqual(k1, k2)
        self.assertEqual(len({k1, k2}), 1)

    def test_key_distinguishes_real_differences(self):
        self.assertNotEqual(app.canonical_partition_key([1, 2, 3], [4, 5, 6]),
                            app.canonical_partition_key([1, 2, 4], [3, 5, 6]))

    def test_partition_difference_accounts_for_the_mirror(self):
        k1 = app.canonical_partition_key([1, 2, 3], [4, 5, 6])
        k2 = app.canonical_partition_key([4, 5, 6], [1, 2, 3])
        self.assertEqual(app.partition_difference(k1, k2), 0)
        # 3 et 4 ont echange de cote : DEUX points ont bouge, pas un.
        k3 = app.canonical_partition_key([1, 2, 4], [3, 5, 6])
        self.assertEqual(app.partition_difference(k1, k3), 2)
        # un seul point deplace, cardinalites differentes
        k4 = app.canonical_partition_key([1, 2], [3, 4, 5, 6])
        self.assertEqual(app.partition_difference(k1, k4), 1)


# ------------------------------------------------ 3-4 : balayage enrichi

class TestSweepSource(unittest.TestCase):

    def test_03_sweep_yields_several_distinct_memberships(self):
        pts = scattered_points(30)
        idx = deliveries(pts)
        out = app._sweep_membership_candidates(pts, idx, len(idx) // 2, 500)
        keys = {app.canonical_partition_key(ga, gb) for _s, ga, gb in out}
        self.assertGreater(len(keys), 10,
                           "le balayage enrichi doit depasser quelques angles fixes")
        self.assertEqual(len(keys), len(out), "doublons dans la source elle-meme")

    def test_04_sweep_is_deterministic(self):
        pts = scattered_points(24)
        idx = deliveries(pts)
        a = app._sweep_membership_candidates(pts, idx, len(idx) // 2, 200)
        b = app._sweep_membership_candidates(pts, idx, len(idx) // 2, 200)
        self.assertEqual([(s, ga, gb) for s, ga, gb in a],
                         [(s, ga, gb) for s, ga, gb in b])

    def test_sweep_respects_its_budget(self):
        pts = scattered_points(30)
        idx = deliveries(pts)
        out = app._sweep_membership_candidates(pts, idx, len(idx) // 2, 7)
        self.assertLessEqual(len(out), 7)


# ------------------------------------ 5-6 : coupures de l'arbre couvrant

class TestMstCuts(unittest.TestCase):

    def _chain_tree(self, n):
        nodes = list(range(1, n + 1))
        return nodes, [(a, b) for a, b in zip(nodes, nodes[1:])]

    def test_05_cut_near_the_target_is_kept(self):
        nodes, tree = self._chain_tree(12)
        pts = grid_points(12)
        out = app._mst_cut_candidates(pts, nodes, 6, tree, 50)
        sizes = sorted(tuple(sorted((len(ga), len(gb)))) for _s, ga, gb in out)
        self.assertIn((6, 6), sizes, "la coupure centrale doit etre proposee")
        for _s, ga, gb in out:
            self.assertEqual(sorted(ga + gb), nodes, "perte ou doublon")

    def test_06_cut_far_from_the_target_is_rejected(self):
        nodes, tree = self._chain_tree(12)
        pts = grid_points(12)
        out = app._mst_cut_candidates(pts, nodes, 6, tree, 50)
        for _s, ga, gb in out:
            gap = min(abs(len(ga) - 6), abs(len(gb) - 6))
            self.assertLessEqual(gap, max(2, 12 // 6),
                                 "coupure trop desequilibree conservee")
        # la coupure 1 / 11 existe dans l'arbre mais ne doit PAS ressortir
        self.assertNotIn(1, [min(len(ga), len(gb)) for _s, ga, gb in out])

    def test_mst_cut_sides_are_connected_in_the_tree(self):
        nodes, tree = self._chain_tree(12)
        adj = chain_graph(nodes)
        pts = grid_points(12)
        for _s, ga, gb in app._mst_cut_candidates(pts, nodes, 6, tree, 50):
            self.assertTrue(app.is_connected_partition(ga, adj)["connected"])
            self.assertTrue(app.is_connected_partition(gb, adj)["connected"])


# ------------------------------------------- 7-8 : croissance de regions

class TestRegionGrowing(unittest.TestCase):

    def test_07_08_cardinality_and_connectivity(self):
        """Cardinalite exacte pour TOUTES les sorties. Connexite certifiee ici
        meme pour celles etiquetees "region_growing" ; celles etiquetees
        "region_growing_repaired" partent vers la reparation commune."""
        pts = scattered_points(24)
        idx, adj, gmeta, dur, _dist = build(pts)
        target = len(idx) // 2
        out = app._region_growing_candidates(pts, idx, target, adj, dur,
                                             gmeta["tree_edges"], 200)
        self.assertGreater(len(out), 0)
        for source, ga, gb in out:
            self.assertEqual(len(ga), target, "cardinalite exacte violee")
            self.assertEqual(len(gb), len(idx) - target)
            self.assertEqual(set(ga) | set(gb), set(idx), "point perdu")
            self.assertEqual(set(ga) & set(gb), set(), "point duplique")
            if source == "region_growing":
                self.assertTrue(
                    app.is_connected_partition(ga, adj)["connected"],
                    "T1 morcele : la croissance n'ajoute que des voisins")
                self.assertTrue(app.is_connected_partition(gb, adj)["connected"])

    def test_region_growing_is_never_silent_on_a_dead_end_terrain(self):
        """Une source muette se lit a tort comme une source inutile. Sur un
        terrain qui enclave les regions, elle doit rendre des candidates a
        reparer plutot que rien du tout."""
        pts = scattered_points(24)
        idx, adj, gmeta, dur, _ = build(pts)
        out = app._region_growing_candidates(pts, idx, len(idx) // 2, adj, dur,
                                             gmeta["tree_edges"], 200)
        self.assertGreater(len(out), 0, "aucune candidate produite")
        labels = {s for s, _a, _b in out}
        self.assertTrue(labels <= {"region_growing", "region_growing_repaired"},
                        "etiquette de source inattendue : %s" % labels)

    def test_region_growing_is_deterministic(self):
        pts = scattered_points(20)
        idx, adj, gmeta, dur, _ = build(pts)
        target = len(idx) // 2
        a = app._region_growing_candidates(pts, idx, target, adj, dur,
                                           gmeta["tree_edges"], 100)
        b = app._region_growing_candidates(pts, idx, target, adj, dur,
                                           gmeta["tree_edges"], 100)
        self.assertEqual(a, b)

    def test_region_growing_yields_more_than_one_partition(self):
        pts = scattered_points(24)
        idx, adj, gmeta, dur, _ = build(pts)
        out = app._region_growing_candidates(pts, idx, len(idx) // 2, adj, dur,
                                             gmeta["tree_edges"], 200)
        keys = {app.canonical_partition_key(ga, gb) for _s, ga, gb in out}
        self.assertGreater(len(keys), 1,
                           "plusieurs germes et regles doivent diverger")


# ------------------------------------------ 9-10 : two-means et k-medoids

class TestClusteringSources(unittest.TestCase):

    def test_09_multiple_two_means_seeds_are_deterministic(self):
        pts = scattered_points(24)
        idx, adj, gmeta, dur, _ = build(pts)
        target = len(idx) // 2
        a = app._two_means_candidates(pts, idx, target, adj, dur,
                                      gmeta["tree_edges"], 50)
        b = app._two_means_candidates(pts, idx, target, adj, dur,
                                      gmeta["tree_edges"], 50)
        self.assertEqual(a, b)
        for _s, ga, gb in a:
            self.assertEqual(len(ga), target)
            self.assertEqual(set(ga) | set(gb), set(idx))

    def test_two_means_default_seed_behaviour_is_unchanged(self):
        """L'appel historique, sans germes, doit rendre exactement ce qu'il
        rendait avant l'ajout du parametre."""
        pts = scattered_points(20)
        idx = deliveries(pts)
        ga, gb = app._two_means_partition(pts, idx, len(idx) // 2)
        self.assertEqual(len(ga), len(idx) // 2)
        self.assertEqual(set(ga) | set(gb), set(idx))
        self.assertEqual((ga, gb),
                         app._two_means_partition(pts, idx, len(idx) // 2))

    def test_10_kmedoids_uses_the_ors_matrix_without_any_network_call(self):
        pts = scattered_points(20)
        idx, adj, gmeta, dur, _ = build(pts)
        target = len(idx) // 2
        with _NoNetwork():
            out = app._kmedoids_candidates(pts, idx, target, adj, dur,
                                           gmeta["tree_edges"], 50)
        self.assertGreater(len(out), 0)
        for _s, ga, gb in out:
            self.assertEqual(len(ga), target)
            self.assertEqual(set(ga) | set(gb), set(idx))

    def test_kmedoids_needs_the_matrix(self):
        pts = scattered_points(20)
        idx, adj, gmeta, _dur, _ = build(pts)
        self.assertEqual(app._kmedoids_candidates(pts, idx, len(idx) // 2, adj,
                                                  None, gmeta["tree_edges"], 50),
                         [])

    def test_symmetrised_ors_handles_missing_directions(self):
        m = [[0.0, 10.0, float("nan")],
             [20.0, 0.0, float("nan")],
             [float("nan"), float("nan"), 0.0]]
        self.assertEqual(app._symmetrised_ors(m, 0, 1), 15.0)
        self.assertIsNone(app._symmetrised_ors(m, 0, 2))
        m2 = [[0.0, 10.0], [float("nan"), 0.0]]
        self.assertEqual(app._symmetrised_ors(m2, 0, 1), 10.0,
                         "une seule direction finie doit suffire")


# ------------------------------------------ 11 : reparations ORS multiples

class TestRepairVariants(unittest.TestCase):

    def test_11_different_repair_rules_can_yield_different_partitions(self):
        """Une partition volontairement imbriquee sur une chaine : chaque
        regle d'absorption et de deplacement la repare autrement."""
        pts = grid_points(10, cols=10)
        idx = deliveries(pts)
        adj = chain_graph(idx)
        mat = euclid_matrix(pts)
        target = len(idx) // 2
        keys = set()
        for comp in app.CONNECTED_COMPONENT_RULES:
            for move in app.CONNECTED_MOVE_RULES:
                ga, gb, ok, _m = app.repair_to_connected_ex(
                    idx[0::2], idx[1::2], adj, pts, mat, 0, 0, target,
                    component_rule=comp, move_rule=move)
                if ok:
                    self.assertEqual(len(ga), target)
                    self.assertTrue(app.is_connected_partition(ga, adj)["connected"])
                    self.assertTrue(app.is_connected_partition(gb, adj)["connected"])
                    keys.add(app.canonical_partition_key(ga, gb))
        self.assertGreaterEqual(len(keys), 1)

    def test_default_repair_still_matches_the_historic_behaviour(self):
        pts = grid_points(12)
        idx = deliveries(pts)
        adj = chain_graph(idx)
        mat = euclid_matrix(pts)
        target = len(idx) // 2
        legacy = app.repair_to_connected(idx[0::2], idx[1::2], adj, pts, mat,
                                         0, 0, target)
        explicit = app.repair_to_connected_ex(idx[0::2], idx[1::2], adj, pts,
                                              mat, 0, 0, target,
                                              component_rule="smallest",
                                              move_rule="ors_delta")
        self.assertEqual(legacy, explicit)


# ------------------------------------------------- 12-16 : perturbations

def grid_graph(rows, cols):
    """Grille rows x cols, numerotee 1..rows*cols en lignes.

    Une chaine n'admet qu'UNE SEULE partition connexe equilibree : aucun
    echange n'y est jamais valide, ce qui ne teste rien. Une grille en admet
    beaucoup, avec des frontieres en escalier -- exactement le terrain que les
    perturbations sont censees explorer.
    """
    node = lambda r, c: r * cols + c + 1          # noqa: E731
    adj = {node(r, c): set() for r in range(rows) for c in range(cols)}
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                adj[node(r, c)].add(node(r, c + 1))
                adj[node(r, c + 1)].add(node(r, c))
            if r + 1 < rows:
                adj[node(r, c)].add(node(r + 1, c))
                adj[node(r + 1, c)].add(node(r, c))
    return adj


class TestPerturbations(unittest.TestCase):
    """Grille 3 x 6, coupee en deux moities de 9 points par la colonne 3.

    ga = colonnes 0-2 = {1,2,3, 7,8,9, 13,14,15}
    gb = colonnes 3-5 = {4,5,6, 10,11,12, 16,17,18}
    """

    def setUp(self):
        self.rows, self.cols = 3, 6
        self.nodes = list(range(1, self.rows * self.cols + 1))
        self.adj = grid_graph(self.rows, self.cols)
        self.ga = sorted(r * self.cols + c + 1
                         for r in range(self.rows) for c in range(3))
        self.gb = sorted(x for x in self.nodes if x not in self.ga)
        self.target = len(self.ga)

    def _swap(self, out_a, in_a):
        na = sorted([x for x in self.ga if x not in out_a] + list(in_a))
        nb = sorted([x for x in self.gb if x not in in_a] + list(out_a))
        return na, nb

    def _variants(self, budget=2000):
        return app._perturbation_candidates([(self.ga, self.gb)], self.nodes,
                                            self.target, self.adj, budget)

    def _accepted(self, budget=2000):
        out = []
        for _s, ga, gb in self._variants(budget):
            ok, _reason, _ = app.validate_partition(ga, gb, self.nodes,
                                                    self.target, self.adj)
            if ok:
                out.append(app.canonical_partition_key(ga, gb))
        return out

    def test_baseline_partition_is_valid(self):
        ok, reason, _ = app.validate_partition(self.ga, self.gb, self.nodes,
                                               self.target, self.adj)
        self.assertTrue(ok, reason)

    def test_12_a_valid_one_for_one_swap_is_accepted(self):
        """3 <-> 10 : 10 se rattache a 9, et 3 se rattache a 4."""
        na, nb = self._swap((3,), (10,))
        ok, reason, _ = app.validate_partition(na, nb, self.nodes, self.target,
                                               self.adj)
        self.assertTrue(ok, "echange 1 contre 1 valide rejete : " + reason)
        self.assertIn(app.canonical_partition_key(na, nb), self._accepted(),
                      "la perturbation 1 contre 1 n'a pas ete produite")

    def test_12b_a_one_for_one_swap_that_disconnects_is_rejected(self):
        """3 <-> 4 : 4 ne touche plus que des points de l'autre groupe."""
        na, nb = self._swap((3,), (4,))
        ok, reason, _ = app.validate_partition(na, nb, self.nodes, self.target,
                                               self.adj)
        self.assertFalse(ok)
        self.assertEqual(reason, "disconnected")
        self.assertNotIn(app.canonical_partition_key(na, nb), self._accepted())

    def test_13_a_valid_two_for_two_swap_is_accepted(self):
        """{3,9} contre {10,16} : 10 et 16 se raccrochent par 15."""
        na, nb = self._swap((3, 9), (10, 16))
        ok, reason, _ = app.validate_partition(na, nb, self.nodes, self.target,
                                               self.adj)
        self.assertTrue(ok, "echange 2 contre 2 valide rejete : " + reason)
        self.assertIn(app.canonical_partition_key(na, nb), self._accepted(),
                      "la perturbation 2 contre 2 n'a pas ete produite")

    def test_14_a_two_for_two_swap_that_disconnects_is_rejected(self):
        """{3,9} contre {4,10} : le couple entrant forme une ile."""
        na, nb = self._swap((3, 9), (4, 10))
        ok, reason, _ = app.validate_partition(na, nb, self.nodes, self.target,
                                               self.adj)
        self.assertFalse(ok)
        self.assertEqual(reason, "disconnected")
        self.assertNotIn(app.canonical_partition_key(na, nb), self._accepted())

    def test_15_a_valid_border_chain_is_accepted(self):
        """Chaine 3-2-1 cedee contre la chaine 10-16-17."""
        na, nb = self._swap((1, 2, 3), (10, 16, 17))
        ok, reason, _ = app.validate_partition(na, nb, self.nodes, self.target,
                                               self.adj)
        self.assertTrue(ok, "chaine frontaliere valide rejetee : " + reason)
        chains = app._border_chains(self.ga, [3, 9, 15], self.adj, 3)
        self.assertIn((3, 2, 1), chains, "la chaine 3-2-1 n'est pas exploree")

    def test_16_a_disconnecting_chain_is_rejected(self):
        """Colonne 2 cedee contre colonne 3 : la colonne entrante est isolee."""
        na, nb = self._swap((3, 9, 15), (4, 10, 16))
        ok, reason, _ = app.validate_partition(na, nb, self.nodes, self.target,
                                               self.adj)
        self.assertFalse(ok)
        self.assertEqual(reason, "disconnected")
        self.assertNotIn(app.canonical_partition_key(na, nb), self._accepted())

    def test_border_chains_stay_inside_the_group_and_are_paths(self):
        for chain in app._border_chains(self.ga, [3, 9, 15], self.adj, 3):
            self.assertEqual(len(set(chain)), 3, "point repete dans la chaine")
            for a, b in zip(chain, chain[1:]):
                self.assertIn(b, self.adj[a], "chaine non adjacente")
            for node in chain:
                self.assertIn(node, self.ga, "chaine sortie du groupe")

    def test_articulation_points_are_pre_filtered(self):
        """Le pre-filtre ecarte les points dont le retrait coupe le groupe.
        Il accelere, il ne decide pas : la connexite est verifiee ensuite."""
        chain = chain_graph([1, 2, 3, 4])
        self.assertTrue(app._is_articulation([1, 2, 3, 4], 2, chain))
        self.assertFalse(app._is_articulation([1, 2, 3, 4], 4, chain))

    def test_every_variant_keeps_the_exact_cardinality(self):
        for _s, ga, gb in self._variants():
            self.assertEqual(len(ga), self.target)
            self.assertEqual(len(gb), len(self.nodes) - self.target)
            self.assertEqual(set(ga) | set(gb), set(self.nodes))
            self.assertEqual(set(ga) & set(gb), set())

    def test_perturbation_budget_is_respected(self):
        self.assertLessEqual(len(self._variants(budget=5)), 5)


# ------------------------------------------------ 17-18 : graphe hybride

class TestHybridGraph(unittest.TestCase):

    def test_17_ors_neighbours_are_added_to_the_haversine_graph(self):
        """Deux points eloignes a vol d'oiseau mais proches par la route
        doivent devenir voisins : c'est tout l'interet du graphe hybride."""
        pts = grid_points(16, cols=4)
        idx = deliveries(pts)
        dur = euclid_matrix(pts)
        plain, pmeta = app.build_geo_graph(pts, idx, dur_matrix=None)
        # une paire tres eloignee geographiquement, rendue quasi gratuite par
        # la route dans les DEUX sens
        far_a, far_b = idx[0], idx[-1]
        self.assertNotIn(far_b, plain[far_a],
                         "fixture invalide : la paire est deja voisine")
        dur[far_a][far_b] = 0.001
        dur[far_b][far_a] = 0.001
        hybrid, hmeta = app.build_geo_graph(pts, idx, dur_matrix=dur, ors_k=1)
        self.assertIn(far_b, hybrid[far_a],
                      "le voisin routier ORS n'a pas ete ajoute")
        self.assertIn(far_a, hybrid[far_b], "arete non symetrique")
        self.assertEqual(hmeta["method"], "knn_haversine_ors_mst")
        self.assertEqual(pmeta["method"], "knn_haversine_mst")
        self.assertGreater(hmeta["ors_edges"], 0)
        self.assertGreater(hmeta["edges"], pmeta["edges"])

    def test_hybrid_graph_only_adds_edges(self):
        pts = scattered_points(20)
        idx = deliveries(pts)
        dur = euclid_matrix(pts)
        plain, _ = app.build_geo_graph(pts, idx, dur_matrix=None)
        hybrid, _ = app.build_geo_graph(pts, idx, dur_matrix=dur, ors_k=3)
        for i in idx:
            self.assertTrue(plain[i] <= hybrid[i],
                            "une arete de securite a ete supprimee")

    def test_18_spanning_tree_alone_keeps_the_graph_connected(self):
        """L'arbre couvrant est le filet de securite : a lui seul il relie
        deja tous les points, meme un point tres isole."""
        pts = scattered_points(24)
        pts.append(P("LOIN", 46.9, 6.4))          # point tres a l'ecart
        idx = deliveries(pts)
        dur = euclid_matrix(pts)
        adj, meta = app.build_geo_graph(pts, idx, dur_matrix=dur)
        self.assertTrue(meta["connected"])
        tree_only = {i: set() for i in idx}
        for u, v in meta["tree_edges"]:
            tree_only[u].add(v)
            tree_only[v].add(u)
        self.assertTrue(app._graph_connected(idx, tree_only),
                        "l'arbre couvrant ne relie pas tous les points")
        self.assertEqual(len(meta["tree_edges"]), len(idx) - 1)

    def test_ors_neighbour_k_zero_falls_back_to_haversine(self):
        pts = scattered_points(16)
        idx = deliveries(pts)
        dur = euclid_matrix(pts)
        _adj, meta = app.build_geo_graph(pts, idx, dur_matrix=dur, ors_k=0)
        self.assertEqual(meta["method"], "knn_haversine_mst")
        self.assertEqual(meta["ors_edges"], 0)


# ---------------------------------------- 19-20 : diversite des finalistes

def _fake_scored(specs):
    """specs : (group_a, group_b, duration, seed)."""
    out = []
    for ga, gb, dur, seed in specs:
        out.append({
            "group_a": list(ga), "group_b": list(gb), "seed": seed,
            "partition_key": app.canonical_partition_key(ga, gb),
            "connected": True, "cardinality_ok": True, "components_total": 0,
            "duration_s": float(dur), "distance_m": float(dur) * 10.0,
            "boundary": {"cut_edges": 1, "enclave_points": 0},
        })
    return out


class TestDiverseFinalists(unittest.TestCase):

    def test_19_selection_is_not_a_dozen_clones(self):
        """Onze quasi-jumelles et deux decoupes franchement differentes : les
        finalistes doivent contenir autre chose que la meme partition."""
        universe = list(range(1, 21))
        specs = []
        for k in range(11):                       # variantes a un point pres
            ga = [x for x in universe[:10] if x != 10] + [11 + k % 9]
            gb = [x for x in universe if x not in ga]
            specs.append((sorted(ga), sorted(gb), 1000 + k, "sweep"))
        specs.append((universe[0::2], universe[1::2], 1400, "mst"))
        specs.append((universe[:10][::-1], universe[10:], 1500, "region_growing"))
        chosen, min_diff = app.select_diverse_finalists(_fake_scored(specs), 5)
        self.assertEqual(len(chosen), 5)
        self.assertGreater(min_diff, 0, "deux finalistes identiques")
        self.assertGreater(len({c["seed"] for c in chosen}), 1,
                           "une seule source represente parmi les finalistes")

    def test_20_the_best_candidate_is_never_dropped(self):
        universe = list(range(1, 21))
        specs = [(universe[:10], universe[10:], 500, "ors_repair")]
        for k in range(15):
            ga = sorted([x for x in universe[:10] if x != 10] + [11 + k % 9])
            gb = sorted(x for x in universe if x not in ga)
            specs.append((ga, gb, 900 + k, "sweep"))
        scored = _fake_scored(specs)
        best = min(scored, key=app._selection_key)
        chosen, _ = app.select_diverse_finalists(scored, 4)
        self.assertIn(best["partition_key"], [c["partition_key"] for c in chosen],
                      "la meilleure candidate a ete sacrifiee a la diversite")

    def test_selection_is_deterministic_and_capped(self):
        universe = list(range(1, 21))
        specs = []
        for k in range(20):
            ga = sorted([x for x in universe[:10] if x != 10] + [11 + k % 9])
            gb = sorted(x for x in universe if x not in ga)
            specs.append((ga, gb, 900 + k, "sweep" if k % 2 else "mst"))
        scored = _fake_scored(specs)
        a, da = app.select_diverse_finalists(scored, 6)
        b, db = app.select_diverse_finalists(scored, 6)
        self.assertEqual([c["partition_key"] for c in a],
                         [c["partition_key"] for c in b])
        self.assertEqual(da, db)
        self.assertLessEqual(len(a), 6)

    def test_fewer_candidates_than_slots_returns_them_all(self):
        universe = list(range(1, 11))
        specs = [(universe[:5], universe[5:], 100, "mst")]
        chosen, _ = app.select_diverse_finalists(_fake_scored(specs), 12)
        self.assertEqual(len(chosen), 1)


# ------------------------------- 21-27 : pipeline complet de generation

class TestGenerationPipeline(unittest.TestCase):

    def _generate(self, n_points):
        pts = scattered_points(n_points)
        idx, adj, gmeta, dur, _dist = build(pts)
        hav = app._build_haversine_matrix(pts)
        target = len(idx) // 2
        with _NoNetwork():
            cands, stats = app.generate_connected_candidates(
                pts, idx, target, adj, dur, hav, 0, 0,
                tree_edges=gmeta["tree_edges"])
        return pts, idx, adj, target, cands, stats

    def test_22_no_network_call_during_generation(self):
        """_NoNetwork fait lever une AssertionError sur tout appel : si la
        generation en tente un, ce test echoue."""
        _p, _i, _a, _t, cands, _s = self._generate(30)
        self.assertGreater(len(cands), 0)

    def test_21_budgets_are_respected(self):
        _p, _i, _a, _t, cands, stats = self._generate(40)
        self.assertLessEqual(len(cands), app.CONNECTED_MAX_UNIQUE_CANDIDATES)
        self.assertLessEqual(stats["raw"],
                             app.CONNECTED_MAX_RAW_CANDIDATES
                             + app.CONNECTED_MAX_PERTURBATIONS)
        self.assertLessEqual(stats["perturbations"],
                             app.CONNECTED_MAX_PERTURBATIONS)

    def test_23_generation_is_deterministic(self):
        a = self._generate(30)[4]
        b = self._generate(30)[4]
        self.assertEqual([c["partition_key"] for c in a],
                         [c["partition_key"] for c in b])
        self.assertEqual([c["seed"] for c in a], [c["seed"] for c in b])

    def test_24_fifty_eight_deliveries_split_29_29(self):
        _p, idx, adj, target, cands, _s = self._generate(58)
        self.assertEqual(target, 29)
        for c in cands:
            self.assertEqual([len(c["group_a"]), len(c["group_b"])], [29, 29])

    def test_25_sixty_deliveries_split_30_30(self):
        _p, idx, adj, target, cands, _s = self._generate(60)
        self.assertEqual(target, 30)
        for c in cands:
            self.assertEqual([len(c["group_a"]), len(c["group_b"])], [30, 30])

    def test_26_27_no_loss_no_duplicate_and_one_component_each(self):
        _p, idx, adj, target, cands, _s = self._generate(40)
        for c in cands:
            ga, gb = c["group_a"], c["group_b"]
            self.assertEqual(set(ga) | set(gb), set(idx), "point perdu")
            self.assertEqual(set(ga) & set(gb), set(), "point duplique")
            self.assertEqual(
                app.is_connected_partition(ga, adj)["component_count"], 1)
            self.assertEqual(
                app.is_connected_partition(gb, adj)["component_count"], 1)

    def test_candidates_are_canonically_unique(self):
        _p, _i, _a, _t, cands, stats = self._generate(40)
        keys = [c["partition_key"] for c in cands]
        self.assertEqual(len(keys), len(set(keys)), "doublon canonique")
        self.assertEqual(stats["unique"], len(cands))

    def test_generation_beats_the_old_handful_of_candidates(self):
        """L'objectif du lot : depasser largement les quelques candidates que
        produisait la generation d'origine."""
        _p, _i, _a, _t, cands, stats = self._generate(60)
        self.assertGreaterEqual(len(cands), 30,
                                "diversification insuffisante : %d uniques"
                                % len(cands))
        self.assertGreaterEqual(len(stats["by_source"]), 3,
                                "trop peu de sources representees : %s"
                                % stats["by_source"])

    def test_no_delivery_point_is_ever_a_depot(self):
        pts = scattered_points(20)
        idx, adj, gmeta, dur, _ = build(pts)
        hav = app._build_haversine_matrix(pts)
        cands, _ = app.generate_connected_candidates(
            pts, idx, len(idx) // 2, adj, dur, hav, 0, 0,
            tree_edges=gmeta["tree_edges"])
        for c in cands:
            self.assertNotIn(0, c["group_a"], "le depart est dans la partition")
            self.assertNotIn(0, c["group_b"])


# --------------------------------- 28-32 : non-regression des autres lots

class TestNoRegression(unittest.TestCase):

    def setUp(self):
        with open("app.py", encoding="utf-8") as fh:
            self.src = fh.read()

    def test_28_kmeans_unchanged(self):
        body = self.src[self.src.index("def kmeans_partition("):
                        self.src.index("def _build_haversine_matrix(")]
        self.assertNotIn("connected", body)
        self.assertNotIn("canonical_partition_key", body)
        for name in ("kmeans_partition", "_balance_groups", "_create_sub_clusters"):
            self.assertIn("def " + name + "(", self.src)

    def test_29_ortools_haversine_unchanged(self):
        body = self.src[self.src.index("def ortools_partition_haversine"):
                        self.src.index("# 4c. MATRICE ORS COMPLETE")]
        self.assertIn("_solve_cvrp_ortools(", body)
        self.assertNotIn("connected", body)

    def test_30_sweep_line_projection_unchanged(self):
        self.assertIn('"territorial_method": "sweep_line_projection"', self.src)
        body = self.src[self.src.index("def ortools_partition_ors_matrix("):
                        self.src.index("# 4e. PARTITION CONNEXE")]
        self.assertNotIn("build_geo_graph", body)
        self.assertNotIn("canonical_partition_key", body)
        self.assertNotIn("select_diverse_finalists", body)

    def test_30b_territorial_enumeration_still_uses_its_own_key(self):
        body = self.src[self.src.index("def enumerate_territorial_partitions("):
                        self.src.index("def _nn_route_matrix(")]
        self.assertIn("_partition_key(", body)
        self.assertNotIn("canonical_partition_key(", body)

    def test_31_the_sequencer_fix_is_still_in_place(self):
        self.assertIn("def select_best_solution(", self.src)
        self.assertIn('meta["connected_routes"]', self.src)
        self.assertIn("presequenced_routes", self.src)
        body = self.src[self.src.index("def select_best_solution("):
                        self.src.index("_SELECTION_REASONS = {")]
        self.assertIn("best_duration + tie_seconds", body)

    def test_32_vroom_budget_is_untouched_by_diversification(self):
        self.assertEqual(app.CONNECTED_VROOM_FINALISTS, 3)
        self.assertEqual(app.CONNECTED_TOP_VROOM, app.CONNECTED_VROOM_FINALISTS)
        self.assertEqual(app.CONNECTED_ORTOOLS_FINALISTS, 12)
        self.assertEqual(app.CONNECTED_TOP_ORTOOLS, app.CONNECTED_ORTOOLS_FINALISTS)
        # la generation ne doit contenir aucun appel a Vroom
        start = self.src.index("# 4f. SOURCES DE PARTITIONS CONNEXES")
        end = self.src.index("def select_diverse_finalists(")
        block = self.src[start:end]
        for forbidden in ("_post_matrix", "_post_vroom", "_resequence_single",
                          "_build_full_matrix_chunked", "_fetch_ors_matrix",
                          "requests."):
            self.assertNotIn(forbidden, block,
                             "appel reseau dans la generation : " + forbidden)

    def test_budgets_are_env_configurable(self):
        for name in ("CONNECTED_TARGET_UNIQUE_CANDIDATES",
                     "CONNECTED_MAX_UNIQUE_CANDIDATES",
                     "CONNECTED_MAX_RAW_CANDIDATES",
                     "CONNECTED_MAX_PERTURBATIONS",
                     "CONNECTED_MAX_CHAIN_LENGTH",
                     "CONNECTED_ORS_NEIGHBOR_K",
                     "CONNECTED_ORTOOLS_FINALISTS",
                     "CONNECTED_VROOM_FINALISTS"):
            self.assertIn('_env_int("%s"' % name, self.src)
            self.assertIsInstance(getattr(app, name), int)

    def test_env_int_rejects_out_of_range_and_garbage(self):
        import os
        os.environ["CONNECTED_TEST_KNOB"] = "not-a-number"
        self.assertEqual(app._env_int("CONNECTED_TEST_KNOB", 7, 1, 10), 7)
        os.environ["CONNECTED_TEST_KNOB"] = "999999"
        self.assertEqual(app._env_int("CONNECTED_TEST_KNOB", 7, 1, 10), 7)
        os.environ["CONNECTED_TEST_KNOB"] = "5"
        self.assertEqual(app._env_int("CONNECTED_TEST_KNOB", 7, 1, 10), 5)
        del os.environ["CONNECTED_TEST_KNOB"]

    def test_no_balancing_objective_anywhere_in_the_connected_pipeline(self):
        body = self.src[self.src.index("def _selection_key"):
                        self.src.index("def _solution_tiebreak")]
        self.assertNotIn("abs(", body)
        body2 = self.src[self.src.index("def select_best_solution("):
                         self.src.index("_SELECTION_REASONS = {")]
        self.assertNotIn("abs(", body2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
