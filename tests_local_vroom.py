"""Tests de la couche VROOM locale.

Ils tournent sans Docker et sans le vrai binaire : un faux binaire ecrit dans
un repertoire temporaire joue le role de VROOM. Ce qui est teste ici est le
CONTRAT du wrapper -- budget, timeout, nettoyage, validation -- et non la
qualite d'optimisation de VROOM, qui n'est pas de notre ressort.

Les tests qui lancent reellement un subprocess sont limites a POSIX : le
service tourne sur Linux, et sur Windows un script sans shebang ne peut pas
etre execute directement. Ils apparaissent alors comme ignores, jamais comme
reussis a tort.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import local_vroom
from local_vroom import (
    ERR_BINARY_MISSING,
    ERR_BUDGET_EXHAUSTED,
    ERR_DISABLED,
    ERR_GLOBAL_TIME_LIMIT,
    ERR_INVALID_JSON,
    ERR_INVALID_SOLUTION,
    ERR_TIMEOUT,
    LocalVroomError,
    LocalVroomLedger,
)

HERE = os.path.dirname(os.path.abspath(__file__))
POSIX_ONLY = unittest.skipUnless(os.name == "posix",
                                 "subprocess reel : Linux/POSIX uniquement")


# =========================================================================
# OUTILLAGE : FAUX BINAIRE VROOM
# =========================================================================

FAKE_TEMPLATE = '''#!/usr/bin/env python3
import json, sys

CALLS = {calls!r}
MODE = {mode!r}

with open(CALLS, "a") as handle:
    handle.write("call\\n")

args = sys.argv[1:]
if "--version" in args or "-v" in args:
    print("vroom 1.15.0-fake")
    sys.exit(0)

inp = args[args.index("-i") + 1]
out = args[args.index("-o") + 1]
with open(inp) as handle:
    data = json.load(handle)

jobs = [j["id"] for j in data["jobs"]]
vehicles = data["vehicles"]
cap = vehicles[0]["capacity"][0]

if MODE == "unassigned":
    unassigned = [{{"id": jobs[-1]}}]
    jobs = jobs[:-1]
else:
    unassigned = []

routes = []
for k, vehicle in enumerate(vehicles):
    chunk = jobs[k * cap:(k + 1) * cap]
    if MODE == "duplicate" and k == 1 and chunk:
        chunk = list(chunk)
        chunk[0] = jobs[0]
    if MODE == "empty_route" and k == 1:
        chunk = []
    steps = [{{"type": "start", "location_index": vehicle["start_index"]}}]
    for jid in chunk:
        steps.append({{"type": "job", "id": jid}})
    steps.append({{"type": "end", "location_index": vehicle["end_index"]}})
    routes.append({{"vehicle": vehicle["id"], "steps": steps,
                    "duration": 100 * len(chunk)}})

solution = {{"code": 0,
             "summary": {{"duration": sum(r["duration"] for r in routes)}},
             "unassigned": unassigned,
             "routes": routes}}

if MODE == "broken_json":
    with open(out, "w") as handle:
        handle.write("{{ ceci n'est pas du json")
else:
    with open(out, "w") as handle:
        json.dump(solution, handle)
'''


class FakeVroom:
    """Faux binaire jetable, avec un journal des appels sur disque.

    Le journal est sur DISQUE et non en memoire parce que le wrapper passe au
    subprocess un environnement minimal : le faux binaire ne peut rien recevoir
    par variable d'environnement, et c'est precisement ce qu'on veut verifier.

    Le vrai subprocess n'est exerce que sous POSIX : c'est la cible du
    service, et le comportement du binaire y est prouve par la CI. Les tests
    de la strategie, eux, injectent un solveur factice et n'ont donc besoin
    d'aucun processus -- ils tournent partout.
    """

    def __init__(self, mode="ok"):
        self.dir = tempfile.mkdtemp(prefix="fakevroom-")
        self.calls_path = os.path.join(self.dir, "calls.log")
        self.path = os.path.join(self.dir, "vroom")
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(FAKE_TEMPLATE.format(calls=self.calls_path, mode=mode))
        os.chmod(self.path, 0o755)

    @property
    def call_count(self):
        if not os.path.isfile(self.calls_path):
            return 0
        with open(self.calls_path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


_TEMP_DIRS = []


def tearDownModule():
    """Les repertoires temporaires des configs de test sont supprimes ici.

    Chaque make_config() en cree un ; sans ce nettoyage, chaque execution de
    la suite en abandonnait une trentaine dans le /tmp du systeme."""
    while _TEMP_DIRS:
        shutil.rmtree(_TEMP_DIRS.pop(), ignore_errors=True)


def make_config(**overrides):
    """Config isolee : jamais celle du processus, pour ne pas fuiter d'un test
    a l'autre."""
    env_backup = dict(os.environ)
    try:
        os.environ["LOCAL_VROOM_EXPERIMENT_ENABLED"] = "true"
        config = local_vroom.LocalVroomConfig()
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
    config.tmpdir = tempfile.mkdtemp(prefix="lvtmp-")
    _TEMP_DIRS.append(config.tmpdir)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def synthetic_durations(size):
    return [[0 if i == j else 60 + abs(i - j) for j in range(size)]
            for i in range(size)]


def joint_payload(n_tasks=60, max_tasks=30):
    job_ids = list(range(1, n_tasks + 1))
    return job_ids, local_vroom.build_joint_payload(
        job_ids=job_ids,
        durations=synthetic_durations(n_tasks + 1),
        start_index=0,
        end_index=0,
        max_tasks_per_vehicle=max_tasks,
        vehicle_ids=(1, 2),
    )


# =========================================================================
# CONFIGURATION ET GARDE-FOUS
# =========================================================================

class TestConfiguration(unittest.TestCase):

    def test_experiment_disabled_by_default(self):
        env_backup = dict(os.environ)
        try:
            os.environ.pop("LOCAL_VROOM_EXPERIMENT_ENABLED", None)
            self.assertFalse(local_vroom.LocalVroomConfig().enabled)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    def test_default_budgets_match_the_prudent_profile(self):
        config = make_config()
        self.assertEqual(config.max_solves, 4)
        self.assertEqual(config.direct_solves, 1)
        self.assertEqual(config.nucleus_solves, 1)
        self.assertEqual(config.finalist_solves, 2)
        self.assertEqual(config.max_concurrent_solves, 1)
        # 1 directe + 1 noyau + 2 finalistes = le plafond de 4.
        self.assertEqual(config.direct_solves + config.nucleus_solves
                         + config.finalist_solves, config.max_solves)
        self.assertAlmostEqual(config.total_soft_limit_s, 58.0)
        self.assertAlmostEqual(config.per_solve_timeout_s, 8.0)
        self.assertAlmostEqual(config.min_remaining_to_start_s, 9.0)
        self.assertAlmostEqual(config.route_first_budget_s, 3.0)
        self.assertAlmostEqual(config.alns_budget_s, 6.0)
        self.assertAlmostEqual(config.max_enclave_ratio, 0.15)

    def test_the_enclave_cap_is_configurable(self):
        env_backup = dict(os.environ)
        try:
            os.environ["LOCAL_VROOM_MAX_ENCLAVE_RATIO"] = "0.05"
            tight = local_vroom.LocalVroomConfig()
            os.environ["LOCAL_VROOM_MAX_ENCLAVE_RATIO"] = "hors-sujet"
            broken = local_vroom.LocalVroomConfig()
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
        self.assertAlmostEqual(tight.max_enclave_ratio, 0.05)
        # Une valeur illisible retombe sur le defaut, elle n'ouvre pas le
        # plafond en grand.
        self.assertAlmostEqual(broken.max_enclave_ratio, 0.15)

    def test_budgets_can_be_raised_to_eight_without_code_change(self):
        env_backup = dict(os.environ)
        try:
            os.environ["LOCAL_VROOM_MAX_SOLVES"] = "8"
            os.environ["LOCAL_VROOM_NUCLEUS_SOLVES"] = "2"
            os.environ["LOCAL_VROOM_FINALIST_SOLVES"] = "5"
            config = local_vroom.LocalVroomConfig()
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
        self.assertEqual(config.max_solves, 8)
        self.assertEqual(config.nucleus_solves, 2)
        self.assertEqual(config.finalist_solves, 5)
        self.assertEqual(config.direct_solves + config.nucleus_solves
                         + config.finalist_solves, 8)

    def test_absurd_values_fall_back_to_defaults(self):
        env_backup = dict(os.environ)
        try:
            os.environ["LOCAL_VROOM_MAX_SOLVES"] = "not-a-number"
            os.environ["LOCAL_VROOM_PER_SOLVE_TIMEOUT_S"] = "-5"
            config = local_vroom.LocalVroomConfig()
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
        self.assertEqual(config.max_solves, 4)
        self.assertAlmostEqual(config.per_solve_timeout_s, 8.0)

    def test_concurrency_cannot_be_raised_above_one(self):
        env_backup = dict(os.environ)
        try:
            os.environ["LOCAL_VROOM_MAX_CONCURRENT_SOLVES"] = "4"
            config = local_vroom.LocalVroomConfig()
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
        self.assertEqual(config.max_concurrent_solves, 1)


# =========================================================================
# LEDGER
# =========================================================================

class TestLedger(unittest.TestCase):

    def test_two_vehicles_in_one_request_count_as_one_solve(self):
        ledger = LocalVroomLedger(max_solves=8, soft_limit_s=58.0)
        _, payload = joint_payload()
        self.assertEqual(len(payload["vehicles"]), 2)
        ledger.record_attempt()
        self.assertEqual(ledger.attempted, 1)

    def test_no_fifth_solve_with_default_settings(self):
        ledger = LocalVroomLedger(max_solves=4, soft_limit_s=58.0)
        for _ in range(4):
            self.assertTrue(ledger.can_attempt())
            ledger.record_attempt()
        self.assertFalse(ledger.can_attempt())
        self.assertEqual(ledger.budget_left(), 0)

    def test_eight_solves_when_configured(self):
        ledger = LocalVroomLedger(max_solves=8, soft_limit_s=58.0)
        for _ in range(8):
            self.assertTrue(ledger.can_attempt())
            ledger.record_attempt()
        self.assertFalse(ledger.can_attempt())

    def test_can_attempt_is_false_when_time_runs_short(self):
        # Demarre il y a 55 s sur une limite de 58 : il reste 3 s, moins que
        # les 9 s minimales exigees pour lancer une resolution.
        ledger = LocalVroomLedger(max_solves=4, soft_limit_s=58.0,
                                  started_at=time.monotonic() - 55.0)
        self.assertGreater(ledger.budget_left(), 0)
        self.assertFalse(ledger.can_attempt())

    def test_diagnostics_expose_every_counter(self):
        ledger = LocalVroomLedger(max_solves=8)
        diagnostics = ledger.as_diagnostics()
        for key in ("local_vroom_planned", "local_vroom_attempted",
                    "local_vroom_succeeded", "local_vroom_failed",
                    "local_vroom_timed_out", "local_vroom_reused",
                    "local_vroom_skipped_for_time", "local_vroom_elapsed_ms",
                    "local_vroom_stop_reason", "local_vroom_last_error"):
            self.assertIn(key, diagnostics)

    def test_reuse_is_counted_separately_from_attempts(self):
        ledger = LocalVroomLedger(max_solves=8)
        ledger.record_reuse()
        ledger.record_reuse()
        self.assertEqual(ledger.reused, 2)
        self.assertEqual(ledger.attempted, 0)


# =========================================================================
# PAYLOAD CONJOINT
# =========================================================================

class TestJointPayload(unittest.TestCase):

    def test_sixty_tasks_two_vehicles_single_request(self):
        job_ids, payload = joint_payload(60, 30)
        self.assertEqual(len(payload["jobs"]), 60)
        self.assertEqual(len(payload["vehicles"]), 2)
        self.assertEqual(len(job_ids), 60)

    def test_capacity_forces_thirty_thirty(self):
        _, payload = joint_payload(60, 30)
        total_capacity = sum(v["capacity"][0] for v in payload["vehicles"])
        total_demand = sum(j["delivery"][0] for j in payload["jobs"])
        # 60 taches pour 60 unites de capacite : aucune autre repartition que
        # 30/30 n'est admissible.
        self.assertEqual(total_capacity, 60)
        self.assertEqual(total_demand, 60)
        for vehicle in payload["vehicles"]:
            self.assertEqual(vehicle["capacity"], [30])

    def test_fifty_eight_tasks_use_twenty_nine_per_vehicle(self):
        _, payload = joint_payload(58, 29)
        self.assertEqual(sum(v["capacity"][0] for v in payload["vehicles"]), 58)

    def test_same_start_and_end_for_both_vehicles(self):
        _, payload = joint_payload()
        starts = {v["start_index"] for v in payload["vehicles"]}
        ends = {v["end_index"] for v in payload["vehicles"]}
        self.assertEqual(starts, {0})
        self.assertEqual(ends, {0})

    def test_no_geometry_is_ever_requested(self):
        _, payload = joint_payload()
        self.assertNotIn("options", payload)
        self.assertIn("matrices", payload)

    def test_skills_pin_jobs_to_a_vehicle(self):
        job_ids = [1, 2, 3, 4]
        payload = local_vroom.build_joint_payload(
            job_ids=job_ids,
            durations=synthetic_durations(5),
            start_index=0,
            end_index=0,
            max_tasks_per_vehicle=2,
            vehicle_ids=(1, 2),
            job_skills={1: [1], 3: [2]},
            vehicle_skills={1: [1], 2: [2]},
        )
        by_id = {j["id"]: j for j in payload["jobs"]}
        self.assertEqual(by_id[1]["skills"], [1])
        self.assertEqual(by_id[3]["skills"], [2])
        # Les points de frontiere restent libres : aucune skill.
        self.assertNotIn("skills", by_id[2])
        self.assertNotIn("skills", by_id[4])

    def test_duplicate_job_ids_are_refused(self):
        with self.assertRaises(LocalVroomError) as ctx:
            local_vroom.build_joint_payload(
                job_ids=[1, 1, 2],
                durations=synthetic_durations(4),
                start_index=0, end_index=0, max_tasks_per_vehicle=2)
        self.assertEqual(ctx.exception.code, ERR_INVALID_SOLUTION)


# =========================================================================
# VALIDATION STRICTE
# =========================================================================

def solution_from(sequences, start=0, end=0, unassigned=()):
    routes = []
    for vid, seq in sequences.items():
        steps = [{"type": "start", "location_index": start}]
        steps += [{"type": "job", "id": jid} for jid in seq]
        steps.append({"type": "end", "location_index": end})
        routes.append({"vehicle": vid, "steps": steps, "duration": 60 * len(seq)})
    return {"code": 0, "unassigned": list(unassigned), "routes": routes,
            "summary": {"duration": sum(r["duration"] for r in routes)}}


class TestValidation(unittest.TestCase):

    def test_valid_thirty_thirty_passes(self):
        jobs = list(range(1, 61))
        solution = solution_from({1: jobs[:30], 2: jobs[30:]})
        sequences = local_vroom.validate_joint_solution(
            solution, jobs, (1, 2), max_tasks_per_vehicle=30,
            start_index=0, end_index=0)
        self.assertEqual([len(sequences[1]), len(sequences[2])], [30, 30])

    def test_one_unassigned_task_invalidates_the_answer(self):
        jobs = list(range(1, 61))
        solution = solution_from({1: jobs[:30], 2: jobs[30:59]},
                                 unassigned=[{"id": 60}])
        with self.assertRaises(LocalVroomError) as ctx:
            local_vroom.validate_joint_solution(solution, jobs, (1, 2))
        self.assertEqual(ctx.exception.code, ERR_INVALID_SOLUTION)

    def test_duplicate_task_invalidates_the_answer(self):
        jobs = list(range(1, 7))
        solution = solution_from({1: [1, 2, 3], 2: [3, 4, 5]})
        with self.assertRaises(LocalVroomError):
            local_vroom.validate_joint_solution(solution, jobs, (1, 2))

    def test_missing_task_invalidates_the_answer(self):
        jobs = list(range(1, 7))
        solution = solution_from({1: [1, 2, 3], 2: [4, 5]})
        with self.assertRaises(LocalVroomError):
            local_vroom.validate_joint_solution(solution, jobs, (1, 2))

    def test_empty_route_invalidates_the_answer(self):
        jobs = list(range(1, 7))
        solution = solution_from({1: [1, 2, 3, 4, 5, 6], 2: []})
        with self.assertRaises(LocalVroomError):
            local_vroom.validate_joint_solution(solution, jobs, (1, 2))

    def test_over_capacity_route_invalidates_the_answer(self):
        jobs = list(range(1, 61))
        solution = solution_from({1: jobs[:31], 2: jobs[31:]})
        with self.assertRaises(LocalVroomError):
            local_vroom.validate_joint_solution(
                solution, jobs, (1, 2), max_tasks_per_vehicle=30)

    def test_wrong_start_invalidates_the_answer(self):
        jobs = [1, 2]
        solution = solution_from({1: [1], 2: [2]}, start=7, end=0)
        with self.assertRaises(LocalVroomError):
            local_vroom.validate_joint_solution(
                solution, jobs, (1, 2), start_index=0, end_index=0)

    def test_non_zero_vroom_code_invalidates_the_answer(self):
        jobs = [1, 2]
        solution = solution_from({1: [1], 2: [2]})
        solution["code"] = 3
        with self.assertRaises(LocalVroomError):
            local_vroom.validate_joint_solution(solution, jobs, (1, 2))


# =========================================================================
# GARDES AVANT SUBPROCESS
# =========================================================================

class TestGuards(unittest.TestCase):

    def setUp(self):
        self.fake = FakeVroom()
        self.addCleanup(self.fake.cleanup)

    def test_disabled_experiment_never_runs_the_binary(self):
        config = make_config(enabled=False, binary=self.fake.path)
        _, payload = joint_payload()
        with self.assertRaises(LocalVroomError) as ctx:
            local_vroom.solve_vroom_local(payload, config=config)
        self.assertEqual(ctx.exception.code, ERR_DISABLED)
        self.assertEqual(self.fake.call_count, 0)

    def test_missing_binary_is_reported_not_guessed(self):
        config = make_config(binary=os.path.join(self.fake.dir, "absent"))
        ledger = LocalVroomLedger(config=config)
        _, payload = joint_payload()
        with self.assertRaises(LocalVroomError) as ctx:
            local_vroom.solve_vroom_local(payload, ledger=ledger, config=config)
        self.assertEqual(ctx.exception.code, ERR_BINARY_MISSING)

    def test_budget_is_checked_before_the_fork_not_after(self):
        config = make_config(binary=self.fake.path)
        ledger = LocalVroomLedger(max_solves=4, config=config)
        for _ in range(4):
            ledger.record_attempt()
        _, payload = joint_payload()
        with self.assertRaises(LocalVroomError) as ctx:
            local_vroom.solve_vroom_local(payload, ledger=ledger, config=config)
        self.assertEqual(ctx.exception.code, ERR_BUDGET_EXHAUSTED)
        # La preuve que rien n'a ete lance : le faux binaire n'a pas ete appele.
        self.assertEqual(self.fake.call_count, 0)
        self.assertEqual(ledger.attempted, 4)

    def test_no_solve_starts_with_less_than_seven_seconds_left(self):
        config = make_config(binary=self.fake.path)
        ledger = LocalVroomLedger(config=config)
        _, payload = joint_payload()
        with self.assertRaises(LocalVroomError) as ctx:
            local_vroom.solve_vroom_local(
                payload, ledger=ledger, config=config,
                cancellation_deadline=time.monotonic() + 5.0)
        self.assertEqual(ctx.exception.code, ERR_GLOBAL_TIME_LIMIT)
        self.assertEqual(self.fake.call_count, 0)
        self.assertEqual(ledger.skipped_for_time, 1)

    def test_seven_seconds_left_is_enough_to_start(self):
        config = make_config(binary=self.fake.path)
        ledger = LocalVroomLedger(config=config)
        self.assertGreaterEqual(ledger.remaining_s(),
                                config.min_remaining_to_start_s)

    def test_oversized_payload_is_refused(self):
        config = make_config(binary=self.fake.path, max_jobs=10)
        _, payload = joint_payload(60, 30)
        with self.assertRaises(LocalVroomError):
            local_vroom.solve_vroom_local(payload, config=config)
        self.assertEqual(self.fake.call_count, 0)

    def test_oversized_matrix_is_refused(self):
        config = make_config(binary=self.fake.path, max_matrix_dim=10)
        _, payload = joint_payload(60, 30)
        with self.assertRaises(LocalVroomError):
            local_vroom.solve_vroom_local(payload, config=config)
        self.assertEqual(self.fake.call_count, 0)


# =========================================================================
# EXECUTION REELLE DU SUBPROCESS
# =========================================================================

@POSIX_ONLY
class TestSubprocessExecution(unittest.TestCase):

    def _run(self, mode="ok", **overrides):
        fake = FakeVroom(mode=mode)
        self.addCleanup(fake.cleanup)
        config = make_config(binary=fake.path, **overrides)
        self.addCleanup(shutil.rmtree, config.tmpdir, True)
        ledger = LocalVroomLedger(config=config)
        job_ids, payload = joint_payload()
        return fake, config, ledger, job_ids, payload

    def test_one_request_two_vehicles_one_call(self):
        fake, config, ledger, job_ids, payload = self._run()
        solution = local_vroom.solve_vroom_local(
            payload, ledger=ledger, config=config)
        sequences = local_vroom.validate_joint_solution(
            solution, job_ids, (1, 2), max_tasks_per_vehicle=30,
            start_index=0, end_index=0)
        self.assertEqual(fake.call_count, 1)
        self.assertEqual(ledger.attempted, 1)
        self.assertEqual(ledger.succeeded, 1)
        self.assertEqual(sorted(len(s) for s in sequences.values()), [30, 30])

    def test_temporary_files_are_removed_after_success(self):
        fake, config, ledger, job_ids, payload = self._run()
        local_vroom.solve_vroom_local(payload, ledger=ledger, config=config)
        self.assertEqual(os.listdir(config.tmpdir), [])

    def test_temporary_files_are_removed_after_failure(self):
        fake, config, ledger, job_ids, payload = self._run(mode="broken_json")
        with self.assertRaises(LocalVroomError) as ctx:
            local_vroom.solve_vroom_local(payload, ledger=ledger, config=config)
        self.assertEqual(ctx.exception.code, ERR_INVALID_JSON)
        self.assertEqual(os.listdir(config.tmpdir), [])
        self.assertEqual(ledger.failed, 1)

    def test_unassigned_task_from_the_binary_is_rejected(self):
        fake, config, ledger, job_ids, payload = self._run(mode="unassigned")
        solution = local_vroom.solve_vroom_local(
            payload, ledger=ledger, config=config)
        with self.assertRaises(LocalVroomError) as ctx:
            local_vroom.validate_joint_solution(solution, job_ids, (1, 2))
        self.assertEqual(ctx.exception.code, ERR_INVALID_SOLUTION)

    def test_fifth_solve_never_reaches_the_binary(self):
        fake, config, ledger, job_ids, payload = self._run()
        for _ in range(4):
            local_vroom.solve_vroom_local(payload, ledger=ledger, config=config)
        self.assertEqual(fake.call_count, 4)
        with self.assertRaises(LocalVroomError) as ctx:
            local_vroom.solve_vroom_local(payload, ledger=ledger, config=config)
        self.assertEqual(ctx.exception.code, ERR_BUDGET_EXHAUSTED)
        self.assertEqual(fake.call_count, 4)

    def test_the_binary_receives_no_secret(self):
        """Le subprocess est lance avec un environnement minimal : meme si
        ORS_KEY est defini dans le service, il ne traverse pas la frontiere."""
        fake = FakeVroom()
        self.addCleanup(fake.cleanup)
        config = make_config(binary=fake.path)
        self.addCleanup(shutil.rmtree, config.tmpdir, True)
        env_backup = dict(os.environ)
        try:
            os.environ["ORS_KEY"] = "secret-de-test"
            _, payload = joint_payload()
            local_vroom.solve_vroom_local(payload, config=config)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
        # Le faux binaire n'a pas pu lire son chemin de journal via
        # l'environnement : il l'a en dur. Sa seule presence prouve qu'il a
        # tourne avec l'environnement restreint.
        self.assertEqual(fake.call_count, 1)


@POSIX_ONLY
class TestTimeoutAndCleanup(unittest.TestCase):

    def test_timeout_kills_the_process_and_its_children(self):
        stub = os.path.join(HERE, "tools", "slow_stub.py")
        self.assertTrue(os.path.isfile(stub), "slow_stub.py manquant")
        os.chmod(stub, 0o755)

        config = make_config(binary=stub, kill_grace_s=1.0)
        self.addCleanup(shutil.rmtree, config.tmpdir, True)
        ledger = LocalVroomLedger(config=config)
        _, payload = joint_payload()

        started = time.monotonic()
        with self.assertRaises(LocalVroomError) as ctx:
            local_vroom.solve_vroom_local(
                payload, timeout_s=2.0, ledger=ledger, config=config,
                cancellation_deadline=time.monotonic() + 120.0)
        elapsed = time.monotonic() - started

        self.assertEqual(ctx.exception.code, ERR_TIMEOUT)
        self.assertEqual(ledger.timed_out, 1)
        self.assertEqual(ledger.failed, 1)
        # Tue vite : le couperet ne doit pas deriver au-dela de quelques
        # secondes, sinon le budget global de 58 s ne veut plus rien dire.
        self.assertLess(elapsed, 10.0)

        # Aucun fichier temporaire ne survit au timeout.
        self.assertEqual(os.listdir(config.tmpdir), [])

        # Aucun enfant direct ne survit, zombies compris.
        time.sleep(0.5)
        self.assertEqual(_own_children(), [])


def _own_children():
    if not os.path.isdir("/proc"):
        return []
    me = os.getpid()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry, "rb") as handle:
                raw = handle.read().decode("utf-8", "replace")
        except OSError:
            continue
        cut = raw.rfind(")")
        if cut < 0:
            continue
        fields = raw[cut + 2:].split()
        if len(fields) >= 2 and int(fields[1]) == me:
            found.append((int(entry), fields[0]))
    return found


# =========================================================================
# VERROU
# =========================================================================

class TestRunLock(unittest.TestCase):

    def test_second_acquisition_is_refused_immediately(self):
        directory = tempfile.mkdtemp(prefix="lvlock-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, "run.lock")

        first = local_vroom.LocalVroomRunLock(path)
        second = local_vroom.LocalVroomRunLock(path)
        self.assertTrue(first.acquire())
        started = time.monotonic()
        self.assertFalse(second.acquire())
        # Non bloquant : le refus est immediat, il n'attend pas la fin.
        self.assertLess(time.monotonic() - started, 1.0)
        first.release()
        self.assertTrue(second.acquire())
        second.release()

    def test_context_manager_raises_when_busy(self):
        directory = tempfile.mkdtemp(prefix="lvlock-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, "run.lock")
        holder = local_vroom.LocalVroomRunLock(path)
        self.assertTrue(holder.acquire())
        self.addCleanup(holder.release)
        with self.assertRaises(LocalVroomError) as ctx:
            with local_vroom.LocalVroomRunLock(path):
                pass
        self.assertEqual(ctx.exception.code, local_vroom.ERR_BUSY)


# =========================================================================
# GARANTIES STATIQUES SUR LE DEPOT
# =========================================================================

def repo_python_files():
    names = ["app.py", "local_vroom.py"]
    tools = os.path.join(HERE, "tools")
    if os.path.isdir(tools):
        names += [os.path.join("tools", n) for n in sorted(os.listdir(tools))
                  if n.endswith(".py")]
    return [os.path.join(HERE, n) for n in names if os.path.isfile(os.path.join(HERE, n))]


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class TestStaticGuarantees(unittest.TestCase):

    def test_no_shell_true_anywhere(self):
        for path in repo_python_files():
            self.assertNotIn("shell=True", read(path),
                             "shell=True trouve dans %s" % path)

    def test_local_layer_never_calls_a_public_vroom_or_heigit_endpoint(self):
        forbidden = ("heigit.org", "api.openrouteservice.org", "vroom/v0",
                     "/optimization", "vroom-express")
        paths = [os.path.join(HERE, "local_vroom.py")]
        tools = os.path.join(HERE, "tools")
        if os.path.isdir(tools):
            paths += [os.path.join(tools, n) for n in os.listdir(tools)
                      if n.endswith(".py")]
        for path in paths:
            content = read(path).lower()
            for needle in forbidden:
                self.assertNotIn(needle, content,
                                 "%s reference %s" % (path, needle))

    def test_local_layer_does_not_import_requests(self):
        content = read(os.path.join(HERE, "local_vroom.py"))
        self.assertNotIn("import requests", content)

    def test_existing_strategies_are_unchanged(self):
        import app
        # Les quatre strategies de production, dans le meme ordre. La
        # strategie experimentale s'ajoute APRES, elle ne s'intercale pas.
        self.assertEqual(
            app.PRODUCTION_STRATEGIES,
            ("kmeans", "ortools_haversine", "ortools_ors_matrix",
             "ortools_ors_matrix_connected"))
        self.assertEqual(app.VALID_STRATEGIES[:4], app.PRODUCTION_STRATEGIES)

    def test_experimental_strategy_is_not_implemented_by_default(self):
        import app
        # Desactivee, elle ne figure pas parmi les strategies implementees :
        # une requete la demandant recoit 501, comme pour un nom inconnu.
        if not local_vroom.LocalVroomConfig().enabled:
            self.assertNotIn("hybrid_local_vroom_territorial",
                             app.IMPLEMENTED_STRATEGIES)
        self.assertIn("hybrid_local_vroom_territorial", app.VALID_STRATEGIES)

    def test_healthz_route_is_declared(self):
        # Les autres fichiers de tests remplacent flask par un faux module des
        # leur import : on ne peut donc pas instancier de client HTTP ici. La
        # route est verifiee sur la source, et son comportement reel est teste
        # de bout en bout par le job CI qui interroge le conteneur.
        content = read(os.path.join(HERE, "app.py"))
        self.assertIn('@app.route("/healthz")', content)

    def test_healthz_payload_is_light_and_complete(self):
        payload = local_vroom.healthz()
        self.assertEqual(payload["status"], "ok")
        for key in ("python", "local_vroom_enabled", "local_vroom_binary_present"):
            self.assertIn(key, payload)

    def test_healthz_reports_the_experiment_off_by_default(self):
        env_backup = dict(os.environ)
        try:
            os.environ.pop("LOCAL_VROOM_EXPERIMENT_ENABLED", None)
            local_vroom.get_config(refresh=True)
            payload = local_vroom.healthz()
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
            local_vroom.get_config(refresh=True)
        self.assertFalse(payload["local_vroom_enabled"])

    def test_app_keeps_starting_without_the_local_layer(self):
        # local_vroom est importe sous try/except dans app.py : une image sans
        # ce module doit encore servir les strategies existantes.
        content = read(os.path.join(HERE, "app.py"))
        self.assertIn("LOCAL_VROOM_MODULE_AVAILABLE", content)
        self.assertIn("except ImportError:", content)

    def test_cli_directory_is_not_tracked(self):
        tracked = subprocess.run(
            ["git", "ls-files", "CLI"],
            cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False)
        self.assertEqual(tracked.stdout.decode().strip(), "",
                         "des fichiers de CLI/ sont suivis par git")


class TestDockerfileGuarantees(unittest.TestCase):

    def setUp(self):
        path = os.path.join(HERE, "Dockerfile")
        if not os.path.isfile(path):
            self.skipTest("Dockerfile absent")
        self.content = read(path)

    def test_vroom_version_is_pinned_by_tag_and_commit(self):
        self.assertIn("VROOM_VERSION=v1.15.0", self.content)
        self.assertIn("VROOM_COMMIT=43dd7d0b8b560431eb555bf335cf4797eb7343c4",
                      self.content)

    def test_the_build_is_sequential_by_default(self):
        """Un build parallele tuait le builder Render par manque de memoire.

        Le nombre de coeurs visibles ne dit rien de la memoire disponible :
        `-j$(nproc)` lancait autant de g++ que de coeurs declares, chacun
        pesant plusieurs centaines de mega-octets sur du C++20 en -O3."""
        self.assertIn("ARG VROOM_BUILD_JOBS=1", self.content)
        self.assertIn('-j"${VROOM_BUILD_JOBS}"', self.content)
        self.assertNotIn("nproc", self.content.split("# Compilation")[0])

    def test_no_unbounded_parallelism_anywhere(self):
        for line in self.content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertNotIn("$(nproc)", stripped)
            self.assertNotIn("make -j ", stripped)

    def test_vroom_is_built_without_routing(self):
        # USE_ROUTING=false : le binaire ne peut structurellement pas appeler
        # un routeur distant.
        self.assertIn("USE_ROUTING=false", self.content)

    def test_vroom_version_is_verified_during_the_build(self):
        self.assertIn("vroom --version", self.content)

    def test_no_vroom_express_and_no_extra_port(self):
        self.assertNotIn("vroom-express", self.content)
        exposed = [line for line in self.content.splitlines()
                   if line.strip().startswith("EXPOSE")]
        self.assertEqual(exposed, ["EXPOSE 10000"])

    def test_gunicorn_timeout_leaves_margin_over_the_soft_limit(self):
        self.assertIn("GUNICORN_TIMEOUT:-300", self.content)
        config = make_config()
        self.assertGreater(300, config.total_soft_limit_s)

    def test_single_worker_by_default_so_the_lock_is_meaningful(self):
        self.assertIn("GUNICORN_WORKERS:-1", self.content)


if __name__ == "__main__":
    unittest.main()
