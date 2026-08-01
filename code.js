// =========================
// CONSTANTES
// =========================
const API_BASE = "https://tournees-api.onrender.com";

const STRATEGIES = ["kmeans", "ortools_haversine", "ortools_ors_matrix",
                    "ortools_ors_matrix_connected",
                    "hybrid_local_vroom_territorial"];
const DEFAULT_STRATEGY = "kmeans";

// Identifiant backend de la stratégie expérimentale. Le libellé de menu et
// cet identifiant sont liés en UN SEUL endroit : une entrée de menu qui
// enverrait un autre nom produirait un 501 côté API, et surtout une ligne de
// Benchmark étiquetée avec une stratégie qui n'a pas tourné.
const EXP_STRATEGY = "hybrid_local_vroom_territorial";

// Libellé de l'entrée principale du menu. Elle lance la stratégie hybride :
// c'est devenu l'usage quotidien, les autres méthodes ne servent plus qu'à
// la comparaison. Le nom interne de la stratégie, lui, ne change pas.
const MENU_OPTIMISER_LABEL = "Optimiser les tournées";

// Titre du menu racine. « Menu tournées » se repère mieux au milieu des
// menus de Google Sheets, qui sont tous des noms communs.
const MENU_RACINE_LABEL = "Menu tournées";

// Ligne de la feuille "Paramètres" portant la stratégie
const STRATEGY_ROW = 6;

// Libelles du champ "Mode" de la feuille Résultats.
// Indexes par partition_engine, qui est la source la plus precise : sous
// strategy_used = kmeans, il distingue l'affectation faite par Vroom multi
// de celle faite par K-Means.
// Vroom séquence dans les quatre cas ; seul le moteur d'affectation change.
const ENGINE_LABELS = {
  "vroom_multi":        "Vroom (affectation + séquencement)",
  "kmeans_fallback":    "K-Means (affectation) + Vroom (séquencement)",
  "ortools_haversine":  "OR-Tools Haversine (affectation) + Vroom (séquencement)",
  "ortools_ors_matrix": "OR-Tools ORS Matrix (affectation) + Vroom (séquencement)",
  "ortools_ors_matrix_connected":
    "OR-Tools ORS Matrix — territoires connexes (affectation) + Vroom (séquencement)",
  "hybrid_local_vroom_territorial":
    "VROOM local conjoint + ALNS territoriale (affectation + séquencement)"
};

// Repli si partition_engine est absent (backend antérieur au lot 3).
const STRATEGY_LABELS = {
  "kmeans":             "K-Means (affectation) + Vroom (séquencement)",
  "ortools_haversine":  "OR-Tools Haversine (affectation) + Vroom (séquencement)",
  "ortools_ors_matrix": "OR-Tools ORS Matrix (affectation) + Vroom (séquencement)",
  "ortools_ors_matrix_connected":
    "OR-Tools ORS Matrix — territoires connexes (affectation) + Vroom (séquencement)",
  "hybrid_local_vroom_territorial":
    "VROOM local conjoint + ALNS territoriale (affectation + séquencement)"
};

const BENCH_SHEET = "Benchmark";

// Colonnes historiques : ordre et libellés figés, ne jamais déplacer ni renommer.
const BENCH_HEADERS_BASE = [
  "Date", "Stratégie exécutée", "Stratégie demandée", "Nb pts", "Signature jeu", "Nb véh",
  "Km T1", "Km T2", "Km total",
  "Min T1", "Min T2", "Min total",
  "Temps calcul (s)", "Appels API", "Vroom", "Matrix",
  "optimization_path", "Répartition"
];

// Colonnes ajoutées par le lot D-3, strictement à la fin.
const BENCH_HEADERS_D3 = [
  "d3_label",
  "max_swap_candidates", "swap_max_consecutive_fails",
  "swap_candidates_tested", "swaps_accepted",
  "swap_resequence_cache_hits", "swap_resequence_cache_misses",
  "swap_vroom_calls_saved", "swap_stop_reason",
  "ortools_solution_limit_effective", "run_error"
];

// Colonnes du certificat territorial, strictement à la fin.
const BENCH_HEADERS_TERR = [
  "territorial_partition", "territorial_method",
  "territorial_membership_locked", "territorial_side_violations",
  "territorial_separator_angle_deg", "territorial_separator_margin_m",
  "territorial_overlap_status", "territorial_candidates_unique",
  "territorial_candidates_scored", "territorial_fallback_used",
  "territorial_error"
];

// Colonnes du certificat de connexite, strictement a la fin.
const BENCH_HEADERS_CONN = [
  "connected_partition", "connected_method", "connected_membership_locked",
  "connected_target_sizes", "connected_components_t1", "connected_components_t2",
  "connected_component_sizes_t1", "connected_component_sizes_t2",
  "connected_candidates_generated", "connected_candidates_valid",
  "connected_candidates_scored", "connected_candidates_ortools",
  "connected_candidates_vroom", "connected_cut_edges", "connected_cut_length_m",
  "connected_cross_neighbors", "connected_enclave_points",
  "connected_selected_seed", "connected_fallback_used", "connected_error",
  "connected_vroom_calls", "selected_sequencer", "final_selection_reason",
  "ortools_total_duration_s", "ortools_total_distance_m",
  "vroom_total_duration_s", "vroom_total_distance_m", "partition_solver"
];

// Colonnes du recentrage ORS-first, strictement a la fin : reference a
// cardinalite exacte, penalite de connexite, budgets par etape, source
// ORS-first et arbitrage de la post-optimisation. Aucune colonne existante
// n'est deplacee.
const BENCH_HEADERS_ORSFIRST = [
  "connected_ors_reference_available", "connected_ors_reference_duration_s",
  "connected_ors_reference_distance_m", "connected_ors_reference_sizes",
  "connected_ors_reference_components_t1",
  "connected_ors_reference_components_t2",
  "connected_ors_reference_time_limit_hit",
  "connected_ors_reference_fallback_used",
  "connected_ors_reference_solver_status", "connected_ors_reference_solve_ms",
  "connectivity_penalty_duration_s", "connectivity_penalty_distance_m",
  "connectivity_penalty_reliable", "connectivity_penalty_note",
  "connected_stage_timings_text", "connected_stage_budget_exhausted",
  "connected_generation_expired_after",
  "connected_ors_repair_candidates_raw",
  "connected_ors_repair_candidates_unique",
  "connected_ors_repair_reached_ortools",
  "connected_ors_repair_best_proxy_rank", "connected_ors_repair_is_winner",
  "connected_prescore_refined", "connected_prescore_budget_exhausted",
  "connected_winner_proxy_rank_rough", "connected_winner_proxy_rank_refined",
  "connected_winner_ortools_rank",
  "connected_matrix_hash", "connected_per_source_text",
  "post_optimization_kept", "post_optimization_note"
];

// Colonnes de la stratégie expérimentale VROOM local + ALNS territoriale,
// strictement à la fin. Aucune colonne existante n'est déplacée, renommée ni
// supprimée : ces colonnes restent vides pour les quatre stratégies de
// production, qui ne renvoient pas ce bloc de diagnostic.
//
// Toutes ces valeurs proviennent de result.ors_matrix.hybrid, déjà présent
// dans la réponse : aucune route Flask n'a eu besoin de changer.
const BENCH_HEADERS_HYBRID = [
  // contexte du run expérimental
  "hybrid_error", "local_vroom_enabled", "local_vroom_version",
  // juge commun : la seule mesure qui classe les solutions
  "common_rescore_duration_s", "common_rescore_distance_m",
  "common_rescore_matrix_hash",
  // résultat de chaque bloc
  "joint_direct_valid", "joint_direct_duration_s", "joint_direct_sizes",
  "joint_nucleus_attempted", "joint_nucleus_valid",
  "joint_nucleus_best_duration_s",
  "route_first_unique", "route_first_best_duration_s",
  "joint_alns_iterations", "joint_alns_accepted", "joint_alns_seed",
  "joint_alns_best_duration_s",
  "joint_finalists", "joint_finalists_local_vroom_solved",
  "joint_finalists_reused",
  // sélection finale
  "joint_solutions_considered", "joint_selected_source",
  "joint_selected_duration_s", "joint_selected_distance_m",
  "joint_selected_sizes", "joint_selected_components",
  "joint_selected_enclaves",
  // admissibilité territoriale
  "joint_territorial_level", "joint_territorial_max_enclaves",
  "joint_territorial_admissible", "joint_territorial_fallback_used",
  "joint_territorial_fallback_reason", "joint_territorial_thresholds",
  "joint_territorial_level_counts",
  // compteur de résolutions locales
  "local_vroom_max_solves", "local_vroom_attempted", "local_vroom_succeeded",
  "local_vroom_failed", "local_vroom_timed_out", "local_vroom_reused",
  "local_vroom_skipped_for_time", "local_vroom_elapsed_ms",
  "local_vroom_stop_reason", "local_vroom_last_error",
  // discipline temporelle, un temps par bloc réellement renvoyé
  "hybrid_stage_matrix_ms", "hybrid_stage_joint_direct_ms",
  "hybrid_stage_route_first_ms", "hybrid_stage_joint_nucleus_ms",
  "hybrid_stage_joint_alns_ms", "hybrid_stage_alns_refine_ms",
  "hybrid_stage_joint_finalists_ms",
  "hybrid_stage_timings_text", "hybrid_stage_stops_text",
  "hybrid_total_elapsed_ms", "hybrid_soft_limit_reached"
];

const BENCH_HEADERS = BENCH_HEADERS_BASE
  .concat(BENCH_HEADERS_D3)
  .concat(BENCH_HEADERS_TERR)
  .concat(BENCH_HEADERS_CONN)
  .concat(BENCH_HEADERS_ORSFIRST)
  .concat(BENCH_HEADERS_HYBRID);



function setupSheets() {

  const ss = SpreadsheetApp.getActive();

  // --- Feuille Paramètres ---
  let paramSheet = ss.getSheetByName("Paramètres");
  if (!paramSheet) {
    paramSheet = ss.insertSheet("Paramètres");
  }
  paramSheet.clear();

  const paramHeaders = ["Paramètre", "Valeur"];
  const paramData = [
    paramHeaders,
    ["Nombre de véhicules", 2],
    ["Max points par véhicule", 35],
    ["Point de départ (ID)", ""],
    ["Point d'arrivée (ID)", ""],
    ["Stratégie", DEFAULT_STRATEGY]
  ];

  paramSheet.getRange(1, 1, paramData.length, 2).setValues(paramData);

  // Menu déroulant sur la cellule Stratégie
  paramSheet.getRange(STRATEGY_ROW, 2).setDataValidation(strategyValidationRule());

  // Style header
  const paramHeaderRange = paramSheet.getRange(1, 1, 1, 2);
  paramHeaderRange.setBackground("#4a86c8").setFontColor("#ffffff").setFontWeight("bold");
  paramSheet.setColumnWidth(1, 250);
  paramSheet.setColumnWidth(2, 200);

  // Bordures
  paramSheet.getRange(1, 1, paramData.length, 2)
    .setBorder(true, true, true, true, true, true);

  // --- Feuille Horodateurs ---
  let horoSheet = ss.getSheetByName("Horodateurs");
  if (!horoSheet) {
    horoSheet = ss.insertSheet("Horodateurs");
  }

  // Ne clear que si vide (pour ne pas écraser les données existantes)
  if (horoSheet.getLastRow() <= 1) {
    horoSheet.clear();
    const horoHeaders = ["ID", "Adresse", "Latitude", "Longitude", "Sélection"];
    horoSheet.getRange(1, 1, 1, 5).setValues([horoHeaders]);
  }

  const horoHeaderRange = horoSheet.getRange(1, 1, 1, 5);
  horoHeaderRange.setBackground("#6aa84f").setFontColor("#ffffff").setFontWeight("bold");
  horoSheet.setColumnWidth(1, 100);
  horoSheet.setColumnWidth(2, 300);
  horoSheet.setColumnWidth(3, 120);
  horoSheet.setColumnWidth(4, 120);
  horoSheet.setColumnWidth(5, 100);

  // Checkbox dans la colonne Sélection (lignes 2 à 100)
  horoSheet.getRange(2, 5, 99, 1).insertCheckboxes();

  // --- Feuille Résultats ---
  let resSheet = ss.getSheetByName("Résultats");
  if (!resSheet) {
    resSheet = ss.insertSheet("Résultats");
  }
  resSheet.clear();

  const resHeaderRange = resSheet.getRange(1, 1, 1, 1);
  resHeaderRange.setValue("Résultats de l'optimisation");
  resHeaderRange.setBackground("#e06666").setFontColor("#ffffff").setFontWeight("bold");
  resSheet.setColumnWidth(1, 150);
  resSheet.setColumnWidth(2, 600);

  // NB : la feuille "Benchmark" n'est jamais touchée ici, son historique est conservé.

  SpreadsheetApp.getActive().toast("Feuilles créées et mises en page !", "Setup", 3);
}


// =========================
// STRATÉGIE : VALIDATION + MIGRATION
// =========================
function strategyValidationRule() {
  return SpreadsheetApp.newDataValidation()
    .requireValueInList(STRATEGIES, true)
    .setAllowInvalid(false)
    .build();
}


/**
 * Ajoute la ligne "Stratégie" aux feuilles Paramètres déjà existantes.
 * Évite d'avoir à relancer setupSheets(), qui efface Paramètres et Résultats.
 * Idempotent : n'écrase jamais une valeur déjà saisie.
 */
function ensureStrategyCell() {

  const sheet = SpreadsheetApp.getActive().getSheetByName("Paramètres");
  if (!sheet) return;

  if (String(sheet.getRange(STRATEGY_ROW, 1).getValue()).trim() === "") {
    sheet.getRange(STRATEGY_ROW, 1).setValue("Stratégie");
    sheet.getRange(STRATEGY_ROW, 2).setValue(DEFAULT_STRATEGY);
    sheet.getRange(STRATEGY_ROW, 1, 1, 2)
      .setBorder(true, true, true, true, true, true);
  }

  sheet.getRange(STRATEGY_ROW, 2).setDataValidation(strategyValidationRule());
}


// =========================
// LIRE PARAMÈTRES
// =========================
function getParams() {

  const sheet = SpreadsheetApp.getActive().getSheetByName("Paramètres");
  const data = sheet.getRange(2, 2, 5, 1).getValues();

  // Cellule vide (feuille antérieure au lot 2) -> défaut kmeans.
  // Valeur saisie mais inconnue -> erreur explicite : une ligne de Benchmark
  // ne doit jamais porter une étiquette de stratégie fausse.
  const raw = data[4][0] === "" || data[4][0] === null || data[4][0] === undefined
    ? DEFAULT_STRATEGY
    : String(data[4][0]).trim().toLowerCase();

  if (STRATEGIES.indexOf(raw) === -1) {
    throw new Error(
      "Stratégie inconnue en Paramètres!B" + STRATEGY_ROW + " : \"" + raw + "\". " +
      "Valeurs acceptées : " + STRATEGIES.join(", ")
    );
  }

  return {
    num_vehicles: Number(data[0][0]) || 2,
    max_per_vehicle: Number(data[1][0]) || 35,
    start_id: data[2][0] ? String(data[2][0]) : "",
    end_id: data[3][0] ? String(data[3][0]) : "",
    strategy: raw
  };
}


// =========================
// LIRE POINTS
// =========================
function getPoints() {

  const sheet = SpreadsheetApp.getActive().getSheetByName("Horodateurs");
  const data = sheet.getDataRange().getValues();

  let points = [];

  for (let i = 1; i < data.length; i++) {

    const selection = data[i][4];

    if (selection === true || selection === "TRUE") {

      points.push({
        id: String(data[i][0]),
        address: data[i][1],
        lat: Number(data[i][2]),
        lon: Number(data[i][3])
      });

    }
  }

  return points;
}


// =========================
// SÉLECTION PAR ID
// =========================
// Prépare la sélection AVANT l'optimisation. Une seule plage est écrite dans
// tout ce bloc : Horodateurs!E, la colonne des cases à cocher. Les ID, les
// adresses et surtout les coordonnées ne sont jamais touchés, pas plus que
// Paramètres!B4:B5 — les deux dépôts restent pilotés depuis la feuille.

// Nom EXACT du fichier HTML à créer dans l'éditeur Apps Script
// (Fichier > Nouveau > Fichier HTML), sans l'extension.
const SELECTION_HTML_FILE = "SelectionParId";

const HORO_SHEET   = "Horodateurs";
const HORO_COL_ID  = 1;   // A
const HORO_COL_SEL = 5;   // E
const PARAM_SHEET  = "Paramètres";
const PARAM_ROW_START = 4;   // B4 : ID de départ
const PARAM_ROW_END   = 5;   // B5 : ID d'arrivée


/**
 * Lit la feuille Horodateurs une seule fois.
 *
 * Deux lectures de la colonne ID, et c'est délibéré. getValues() rend le type
 * natif : un ID « 0012 » saisi dans une cellule numérique revient 12, et ne
 * correspondrait plus à ce que l'utilisateur colle. getDisplayValues() rend
 * ce qui est AFFICHÉ. On indexe les deux formes, donc les deux fonctionnent,
 * et un écart entre elles est signalé plutôt que subi.
 */
function _lireHorodateurs_() {
  const sheet = SpreadsheetApp.getActive().getSheetByName(HORO_SHEET);
  if (!sheet) throw new Error("Feuille « " + HORO_SHEET + " » introuvable.");

  const lastRow = sheet.getLastRow();
  if (lastRow < 2) return {sheet: sheet, lastRow: lastRow, lignes: [], index: {}};

  const brut = sheet.getRange(2, 1, lastRow - 1, 4).getValues();
  const affiche = sheet.getRange(2, HORO_COL_ID, lastRow - 1, 1).getDisplayValues();

  const lignes = [];
  const index = {};
  for (var i = 0; i < brut.length; i++) {
    const ligne = {
      row: i + 2,
      idBrut: brut[i][0] === null || brut[i][0] === undefined ? "" : String(brut[i][0]).trim(),
      idAffiche: String(affiche[i][0] || "").trim(),
      libelle: String(brut[i][1] || "")
    };
    lignes.push(ligne);
    // Les deux écritures pointent vers la même ligne : coller « 12 » ou
    // « 0012 » trouve la cellule dans les deux cas.
    [ligne.idAffiche, ligne.idBrut].forEach(function (cle) {
      if (!cle) return;
      if (!index[cle]) index[cle] = [];
      if (index[cle].indexOf(ligne.row) === -1) index[cle].push(ligne.row);
    });
  }
  return {sheet: sheet, lastRow: lastRow, lignes: lignes, index: index};
}


/** Résout un ID : {trouve, ambigu, rows, libelle}. Ne choisit jamais seul. */
function _resoudreId_(index, lignes, id) {
  const rows = index[String(id || "").trim()] || [];
  const parRow = {};
  for (var i = 0; i < lignes.length; i++) parRow[lignes[i].row] = lignes[i];
  return {
    id: String(id || "").trim(),
    trouve: rows.length === 1,
    ambigu: rows.length > 1,
    rows: rows,
    libelle: rows.length ? (parRow[rows[0]].libelle || "") : ""
  };
}


/**
 * Découpe la saisie libre en ID.
 *
 * Séparateurs : retour à la ligne, virgule, point-virgule, tabulation — soit
 * un collage direct depuis une colonne de tableur. L'espace n'en est PAS un :
 * un identifiant peut légitimement en contenir. La normalisation se limite au
 * rognage des bords, au retrait des vides et au dédoublonnage exact. Aucun ID
 * n'est converti : « 0012 » reste « 0012 ».
 */
function _parseIds_(texte) {
  const bruts = String(texte || "").split(/[\n\r,;\t]+/);
  const vus = {};
  const uniques = [];
  var saisis = 0;
  for (var i = 0; i < bruts.length; i++) {
    const id = bruts[i].trim();
    if (!id) continue;
    saisis++;
    if (vus[id]) continue;
    vus[id] = true;
    uniques.push(id);
  }
  return {saisis: saisis, uniques: uniques};
}


/**
 * Contexte affiché à l'ouverture de la barre latérale.
 *
 * Lecture SEULE de Paramètres!B4 et B5 : la barre latérale montre les deux
 * dépôts, elle ne les modifie jamais. Les corriger reste une action dans la
 * feuille Paramètres.
 */
function getSelectionContexte() {
  try {
    const param = SpreadsheetApp.getActive().getSheetByName(PARAM_SHEET);
    if (!param) throw new Error("Feuille « " + PARAM_SHEET + " » introuvable.");

    const startId = String(param.getRange(PARAM_ROW_START, 2).getValue() || "").trim();
    const endId = String(param.getRange(PARAM_ROW_END, 2).getValue() || "").trim();

    const data = _lireHorodateurs_();
    const depart = _resoudreId_(data.index, data.lignes, startId);
    const arrivee = endId ? _resoudreId_(data.index, data.lignes, endId) : null;

    return JSON.stringify({
      depart: {id: startId, libelle: depart.libelle, trouve: depart.trouve,
               ambigu: depart.ambigu, vide: !startId},
      arrivee: arrivee
        ? {id: endId, libelle: arrivee.libelle, trouve: arrivee.trouve,
           ambigu: arrivee.ambigu, vide: false}
        : {id: startId, libelle: depart.libelle, trouve: depart.trouve,
           ambigu: depart.ambigu, vide: true},
      memeDepot: !endId || endId === startId,
      nbLignes: data.lignes.length
    });
  } catch (e) {
    return JSON.stringify({erreur: String(e && e.message ? e.message : e)});
  }
}


/**
 * Valide la saisie et applique la sélection.
 *
 * `appliquer` à false ne fait que rapporter : rien n'est écrit tant que
 * l'utilisateur n'a pas vu le bilan. Un dépôt vide, introuvable ou ambigu
 * bloque dans les deux cas.
 */
function appliquerSelectionParId(texteCollectes, appliquer) {
  try {
    const param = SpreadsheetApp.getActive().getSheetByName(PARAM_SHEET);
    if (!param) throw new Error("Feuille « " + PARAM_SHEET + " » introuvable.");

    const startId = String(param.getRange(PARAM_ROW_START, 2).getValue() || "").trim();
    const endIdBrut = String(param.getRange(PARAM_ROW_END, 2).getValue() || "").trim();
    const endId = endIdBrut || startId;

    const data = _lireHorodateurs_();
    const rapport = {
      applique: false, bloquant: null,
      depart: null, arrivee: null,
      saisis: 0, uniques: 0, trouves: 0,
      inconnus: [], ambigus: [], depotsDansCollectes: [],
      lignesCochees: 0, collectesCochees: 0
    };

    // --- dépôts : bloquants, et corrigés dans la feuille, pas ici ---
    if (!startId) {
      rapport.bloquant = "Aucun point de départ n'est configuré dans "
        + PARAM_SHEET + "!B" + PARAM_ROW_START
        + ". Renseignez-le avant de sélectionner les collectes.";
      return JSON.stringify(rapport);
    }
    const depart = _resoudreId_(data.index, data.lignes, startId);
    const arrivee = _resoudreId_(data.index, data.lignes, endId);
    rapport.depart = {id: startId, libelle: depart.libelle,
                      trouve: depart.trouve, ambigu: depart.ambigu};
    rapport.arrivee = {id: endId, libelle: arrivee.libelle,
                       trouve: arrivee.trouve, ambigu: arrivee.ambigu};

    if (!depart.trouve) {
      rapport.bloquant = depart.ambigu
        ? ("Le point de départ « " + startId + " » correspond à plusieurs lignes ("
           + depart.rows.join(", ") + "). Corrigez la feuille " + HORO_SHEET + ".")
        : ("Le point de départ configuré dans " + PARAM_SHEET + "!B" + PARAM_ROW_START
           + " est introuvable. Corrigez la feuille " + PARAM_SHEET
           + " avant de sélectionner les collectes.");
      return JSON.stringify(rapport);
    }
    if (!arrivee.trouve) {
      rapport.bloquant = arrivee.ambigu
        ? ("Le point d'arrivée « " + endId + " » correspond à plusieurs lignes ("
           + arrivee.rows.join(", ") + "). Corrigez la feuille " + HORO_SHEET + ".")
        : ("Le point d'arrivée configuré dans " + PARAM_SHEET + "!B" + PARAM_ROW_END
           + " est introuvable. Corrigez la feuille " + PARAM_SHEET
           + " avant de sélectionner les collectes.");
      return JSON.stringify(rapport);
    }

    // --- collectes ---
    const parsed = _parseIds_(texteCollectes);
    rapport.saisis = parsed.saisis;
    rapport.uniques = parsed.uniques.length;

    const rowsDepots = {};
    rowsDepots[depart.rows[0]] = true;
    rowsDepots[arrivee.rows[0]] = true;   // même ligne si départ = arrivée

    const rowsCollectes = [];
    for (var i = 0; i < parsed.uniques.length; i++) {
      const id = parsed.uniques[i];
      const trouve = _resoudreId_(data.index, data.lignes, id);
      if (trouve.ambigu) { rapport.ambigus.push({id: id, rows: trouve.rows}); continue; }
      if (!trouve.trouve) { rapport.inconnus.push(id); continue; }
      if (rowsDepots[trouve.rows[0]]) { rapport.depotsDansCollectes.push(id); continue; }
      rapport.trouves++;
      rowsCollectes.push(trouve.rows[0]);
    }

    if (!parsed.uniques.length) {
      rapport.bloquant = "Aucun identifiant de collecte n'a été saisi.";
      return JSON.stringify(rapport);
    }

    const nbDepots = Object.keys(rowsDepots).length;
    rapport.collectesCochees = rowsCollectes.length;
    rapport.lignesCochees = rowsCollectes.length + nbDepots;
    rapport.nbDepots = nbDepots;

    if (!appliquer) return JSON.stringify(rapport);

    // Rien n'est appliqué tant qu'un ID reste inconnu ou ambigu : les ignorer
    // en silence produirait une tournée incomplète sans que personne ne le voie.
    if (rapport.inconnus.length || rapport.ambigus.length) {
      rapport.bloquant = "Des identifiants sont inconnus ou ambigus. "
        + "Corrigez la saisie, ou confirmez explicitement.";
      return JSON.stringify(rapport);
    }

    // --- UNIQUE écriture de tout ce bloc ---
    // Une seule plage, une seule setValues() : la colonne des cases. Aucune
    // écriture ligne par ligne, aucune autre colonne.
    const aCocher = {};
    Object.keys(rowsDepots).forEach(function (r) { aCocher[r] = true; });
    rowsCollectes.forEach(function (r) { aCocher[r] = true; });

    const valeurs = [];
    for (var r = 2; r <= data.lastRow; r++) valeurs.push([!!aCocher[r]]);
    data.sheet.getRange(2, HORO_COL_SEL, valeurs.length, 1).setValues(valeurs);

    rapport.applique = true;
    return JSON.stringify(rapport);

  } catch (e) {
    return JSON.stringify({applique: false,
                           bloquant: String(e && e.message ? e.message : e)});
  }
}


/** Entrée de menu : ouvre la barre latérale de sélection. */
function ouvrirSelectionParId() {
  const html = HtmlService.createHtmlOutputFromFile(SELECTION_HTML_FILE)
    .setTitle("Sélectionner les points");
  SpreadsheetApp.getUi().showSidebar(html);
}


// =========================
// APPEL API
// =========================
function callAPI(points, params) {

  const url = API_BASE + "/optimize";

  const payload = {
    points: points,
    num_vehicles: params.num_vehicles,
    max_per_vehicle: params.max_per_vehicle,
    start_id: params.start_id,
    end_id: params.end_id,
    strategy: params.strategy
  };

  // Réveil du serveur (Render free tier s'endort, peut prendre 30-60s)
  for (var attempt = 0; attempt < 6; attempt++) {
    try {
      var wakeResp = UrlFetchApp.fetch(API_BASE + "/", { muteHttpExceptions: true });
      if (wakeResp.getResponseCode() === 200) break;
    } catch (e) {}
    Utilities.sleep(10000);
  }

  const response = UrlFetchApp.fetch(url, {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  });

  const text = response.getContentText();
  const code = response.getResponseCode();

  // Le backend renvoie un corps JSON explicite sur 400 (stratégie inconnue)
  // et 501 (stratégie pas encore implémentée). Sans ce traitement, ces cas
  // étaient masqués par le message générique "L'API n'est pas prête".
  if (code !== 200) {
    var detail = "";
    try {
      var errJson = JSON.parse(text);
      if (errJson && errJson.error) detail = String(errJson.error);
    } catch (e) {}

    if (detail) {
      throw new Error("API (code " + code + ") : " + detail);
    }
    throw new Error("L'API n'est pas prête (code " + code + "). Réessayez dans 1 minute.");
  }

  if (text.startsWith("<!")) {
    throw new Error("L'API n'est pas prête (réponse HTML). Réessayez dans 1 minute.");
  }

  return JSON.parse(text);
}


// =========================
// LIBELLÉ DU MODE
// =========================
/**
 * Construit le texte de la cellule "Mode" de la feuille Résultats.
 *
 * L'ancienne version se déduisait uniquement de vroom_used, ce qui produisait
 * deux libellés faux depuis l'arrivée du sélecteur de stratégie :
 *   - vroom_used = false affichait "K-Means (affectation)" quelle que soit la
 *     stratégie, y compris sous ortools_haversine / ortools_ors_matrix ;
 *   - vroom_used = true affichait "Vroom (affectation + séquencement)" alors
 *     que sur 62 points l'affectation vient de K-Means.
 *
 * Le libellé ne suppose donc plus rien : il vient de partition_engine, sinon
 * de strategy_used, et reste explicitement indéterminé si aucun des deux n'est
 * renseigné. Un échec de séquencement Vroom est signalé à part, car il ne
 * change jamais la stratégie d'affectation.
 */
function buildModeText(result) {

  var label = ENGINE_LABELS[result.partition_engine]
           || STRATEGY_LABELS[result.strategy_used]
           || "Mode indéterminé (réponse sans partition_engine ni strategy_used)";

  var text = label;

  var steps = result.post_processing;
  if (steps && steps.length) {
    text += " | post-traitement : " + steps.join(", ");
  }

  // === et non !result.vroom_used : une réponse sans le champ ne doit pas
  // être interprétée comme un échec de séquencement.
  if (result.vroom_used === false) {
    text += " | fallback séquencement : " + (result.vroom_error || "raison inconnue");
  }

  return text;
}


// =========================
// ÉCRIRE RÉSULTATS
// =========================
function writeResult(result, params, points) {

  const ss = SpreadsheetApp.getActive();
  let sheet = ss.getSheetByName("Résultats");
  if (!sheet) {
    sheet = ss.insertSheet("Résultats");
  }
  sheet.clear();

  // Dictionnaires ID → Adresse / Lat / Lon
  const addressMap = {};
  const latMap = {};
  const lonMap = {};
  for (let i = 0; i < points.length; i++) {
    const sid = String(points[i].id);
    addressMap[sid] = points[i].address || "";
    latMap[sid] = points[i].lat || "";
    lonMap[sid] = points[i].lon || "";
  }

  // Récupérer les tournées
  const t1 = result["tournee_1"] || [];
  const t2 = result["tournee_2"] || [];
  const maxLen = Math.max(t1.length, t2.length);

  // Largeurs colonnes
  sheet.setColumnWidth(1, 70);   // Ordre
  sheet.setColumnWidth(2, 70);   // T1 ID
  sheet.setColumnWidth(3, 300);  // T1 Adresse
  sheet.setColumnWidth(4, 100);  // T1 Lat
  sheet.setColumnWidth(5, 100);  // T1 Lon
  sheet.setColumnWidth(6, 70);   // T2 ID
  sheet.setColumnWidth(7, 300);  // T2 Adresse
  sheet.setColumnWidth(8, 100);  // T2 Lat
  sheet.setColumnWidth(9, 100);  // T2 Lon

  // --- Ligne 1 : en-têtes principaux ---
  sheet.getRange(1, 1).setValue("Ordre");
  sheet.getRange(1, 1).setBackground("#434343").setFontColor("#ffffff").setFontWeight("bold");

  sheet.getRange(1, 2, 1, 4).merge();
  sheet.getRange(1, 2).setValue("Tournée 1 (" + t1.length + " pts)");
  sheet.getRange(1, 2).setBackground("#6aa84f").setFontColor("#ffffff").setFontWeight("bold");
  sheet.getRange(1, 2).setHorizontalAlignment("center");

  sheet.getRange(1, 6, 1, 4).merge();
  sheet.getRange(1, 6).setValue("Tournée 2 (" + t2.length + " pts)");
  sheet.getRange(1, 6).setBackground("#4a86c8").setFontColor("#ffffff").setFontWeight("bold");
  sheet.getRange(1, 6).setHorizontalAlignment("center");

  // --- Ligne 2 : sous-en-têtes ---
  sheet.getRange(2, 1, 1, 9).setValues([["", "ID", "Adresse", "Lat", "Lon", "ID", "Adresse", "Lat", "Lon"]]);
  sheet.getRange(2, 1, 1, 9).setBackground("#f3f3f3").setFontWeight("bold");

  // --- Lignes de données ---
  if (maxLen > 0) {
    const rows = [];
    for (let i = 0; i < maxLen; i++) {
      const id1 = i < t1.length ? String(t1[i]) : "";
      const addr1 = id1 ? (addressMap[id1] || "") : "";
      const lat1 = id1 ? (latMap[id1] || "") : "";
      const lon1 = id1 ? (lonMap[id1] || "") : "";
      const id2 = i < t2.length ? String(t2[i]) : "";
      const addr2 = id2 ? (addressMap[id2] || "") : "";
      const lat2 = id2 ? (latMap[id2] || "") : "";
      const lon2 = id2 ? (lonMap[id2] || "") : "";
      rows.push([i + 1, id1, addr1, lat1, lon1, id2, addr2, lat2, lon2]);
    }
    sheet.getRange(3, 1, rows.length, 9).setValues(rows);

    // Alternance couleur lignes
    for (let i = 0; i < rows.length; i++) {
      const bg = (i % 2 === 0) ? "#ffffff" : "#f9f9f9";
      sheet.getRange(3 + i, 1, 1, 9).setBackground(bg);
    }
  }

  // Bordures sur tout le tableau
  const totalRows = maxLen + 2;
  sheet.getRange(1, 1, totalRows, 9).setBorder(true, true, true, true, true, true);

  // Séparateur visuel entre T1 et T2
  sheet.getRange(1, 6, totalRows, 1)
    .setBorder(null, true, null, null, null, null, "#000000", SpreadsheetApp.BorderStyle.SOLID_MEDIUM);

  // Info clusters + mode + distances
  const infoRow = totalRows + 2;
  sheet.getRange(infoRow, 1).setValue("Clusters DBSCAN");
  sheet.getRange(infoRow, 2).setValue(result.num_clusters_dbscan || "");
  sheet.getRange(infoRow, 1).setFontWeight("bold");

  var modeText = buildModeText(result);
  sheet.getRange(infoRow + 1, 1).setValue("Mode");
  sheet.getRange(infoRow + 1, 2).setValue(modeText);
  sheet.getRange(infoRow + 1, 1).setFontWeight("bold");

  var km1  = result.tournee_1_km  || 0;
  var km2  = result.tournee_2_km  || 0;
  var min1 = result.tournee_1_min;
  var min2 = result.tournee_2_min;

  var dist1Text = km1 + " km routiers" + (min1 != null ? " (~" + min1 + " min)" : "");
  var dist2Text = km2 + " km routiers" + (min2 != null ? " (~" + min2 + " min)" : "");
  var totalKm   = Math.round((km1 + km2) * 100) / 100;
  var totalMin  = (min1 != null && min2 != null) ? Math.round((min1 + min2) * 10) / 10 : null;
  var totalText = totalKm + " km routiers" + (totalMin != null ? " (~" + totalMin + " min)" : "");

  sheet.getRange(infoRow + 2, 1).setValue("Distance Tournée 1");
  sheet.getRange(infoRow + 2, 2).setValue(dist1Text);
  sheet.getRange(infoRow + 2, 1).setFontWeight("bold");
  sheet.getRange(infoRow + 3, 1).setValue("Distance Tournée 2");
  sheet.getRange(infoRow + 3, 2).setValue(dist2Text);
  sheet.getRange(infoRow + 3, 1).setFontWeight("bold");
  sheet.getRange(infoRow + 4, 1).setValue("Distance totale");
  sheet.getRange(infoRow + 4, 2).setValue(totalText);
  sheet.getRange(infoRow + 4, 1).setFontWeight("bold");
  sheet.getRange(infoRow + 4, 2).setFontWeight("bold");

  SpreadsheetApp.getActive().toast("Optimisation terminée !", "Résultat", 3);
}


// =========================
// BENCHMARK (historique cumulatif, jamais effacé)
// =========================
function _num(v) {
  return (v === null || v === undefined || v === "") ? "" : Number(v);
}


/**
 * Crée la feuille Benchmark si besoin, et complète l'en-tête des feuilles
 * antérieures au lot D-3 en n'écrivant QUE les colonnes manquantes, à la fin.
 * Aucune colonne existante n'est déplacée, renommée ni effacée.
 */
function ensureBenchmarkSheet() {

  const ss = SpreadsheetApp.getActive();
  let sheet = ss.getSheetByName(BENCH_SHEET);

  // Une feuille Apps Script naît avec 26 colonnes ; l'en-tête en compte 29
  // depuis le lot D-3. Sans cet élargissement, getRange et appendRow lèveraient
  // « out of bounds ».
  const widen = function (sh) {
    const missing = BENCH_HEADERS.length - sh.getMaxColumns();
    if (missing > 0) sh.insertColumnsAfter(sh.getMaxColumns(), missing);
  };

  if (!sheet) {
    sheet = ss.insertSheet(BENCH_SHEET);
    widen(sheet);
    sheet.getRange(1, 1, 1, BENCH_HEADERS.length).setValues([BENCH_HEADERS]);
    sheet.getRange(1, 1, 1, BENCH_HEADERS.length)
      .setBackground("#434343").setFontColor("#ffffff").setFontWeight("bold");
    sheet.setFrozenRows(1);
    sheet.setColumnWidth(1, 140);   // Date
    sheet.setColumnWidth(2, 150);   // Stratégie exécutée
    sheet.setColumnWidth(3, 150);   // Stratégie demandée
    sheet.setColumnWidth(17, 240);  // optimization_path
    return sheet;
  }

  // Migration : la feuille existe avec les 18 colonnes historiques.
  widen(sheet);
  const existing = sheet.getRange(1, 1, 1, BENCH_HEADERS.length).getValues()[0];
  const missing = [];
  for (var i = 0; i < BENCH_HEADERS.length; i++) {
    if (String(existing[i] || "").trim() === "") missing.push(i);
  }
  if (missing.length) {
    const first = missing[0];
    const tail = BENCH_HEADERS.slice(first);
    sheet.getRange(1, first + 1, 1, tail.length).setValues([tail]);
    sheet.getRange(1, first + 1, 1, tail.length)
      .setBackground("#434343").setFontColor("#ffffff").setFontWeight("bold");
  }
  return sheet;
}


/**
 * Ajoute une ligne à la feuille "Benchmark". Ne nettoie jamais, ne réécrit
 * jamais les lignes précédentes : c'est l'historique de comparaison.
 *
 * Les métriques de swaps sont lues directement dans la réponse du backend :
 * une optimisation normale les enregistre donc sans dépendre d'un label.
 *
 * @param {Object=} extra  {d3_label, run_error}. Absent ou partiel -> les
 *                         cellules correspondantes restent vides.
 */
function appendBenchmark(result, params, points, extra) {

  const sheet = ensureBenchmarkSheet();
  extra = extra || {};
  result = result || {};

  const km1 = _num(result.tournee_1_km);
  const km2 = _num(result.tournee_2_km);
  const min1 = _num(result.tournee_1_min);
  const min2 = _num(result.tournee_2_min);

  const kmTotal  = (km1 !== "" && km2 !== "")   ? Math.round((km1 + km2) * 100) / 100 : "";
  const minTotal = (min1 !== "" && min2 !== "") ? Math.round((min1 + min2) * 10) / 10 : "";

  const calls = result.api_calls || {};
  const sizes = result.partition_sizes || [];

  // Diagnostic de la stratégie expérimentale. Absent des quatre stratégies de
  // production : hyb vaut alors {} et toutes ses colonnes restent vides.
  const hyb = _hybridDiag(result);
  const stages = hyb.hybrid_stages || [];

  // Les champs D-3 sont absents des réponses d'un backend antérieur :
  // _cell() rend "" plutôt que de lever, les runs normaux restent valides.
  const row = [
    new Date(),
    // strategy_used = ce qui a RÉELLEMENT tourné. Un écart avec la colonne
    // suivante signale un repli et invalide la ligne comme point de comparaison.
    result.strategy_used || "",
    result.strategy_requested || params.strategy,
    points.length,
    result.points_signature || "",
    params.num_vehicles,
    km1, km2, kmTotal,
    min1, min2, minTotal,
    result.elapsed_ms != null ? Math.round(result.elapsed_ms / 100) / 10 : "",
    calls.total  != null ? calls.total  : "",
    calls.vroom  != null ? calls.vroom  : "",
    calls.matrix != null ? calls.matrix : "",
    result.optimization_path || "",
    sizes.join(" / "),

    // --- colonnes D-3 ---
    extra.d3_label || "",
    _cell(result.max_swap_candidates),
    _cell(result.swap_max_consecutive_fails),
    _cell(result.swap_candidates_tested),
    _cell(result.swaps_accepted),
    _cell(result.swap_resequence_cache_hits),
    _cell(result.swap_resequence_cache_misses),
    _cell(result.swap_vroom_calls_saved),
    _cell(result.swap_stop_reason),
    _cell(result.ortools_solution_limit),
    extra.run_error || "",

    // --- certificat territorial ---
    _cell(result.territorial_partition),
    _cell(result.territorial_method),
    _cell(result.territorial_membership_locked),
    _cell(result.territorial_side_violations),
    _cell(result.territorial_separator_angle_deg),
    _cell(result.territorial_separator_margin_m),
    _cell(result.territorial_overlap_status),
    _cell(result.territorial_candidates_unique),
    _cell(result.territorial_candidates_scored),
    _cell(result.territorial_fallback_used),
    _cell(result.territorial_error),

    // --- certificat de connexite ---
    _cell(result.connected_partition),
    _cell(result.connected_method),
    _cell(result.connected_membership_locked),
    _cellList(result.connected_target_sizes),
    _cell(result.connected_components_t1),
    _cell(result.connected_components_t2),
    _cellList(result.connected_component_sizes_t1),
    _cellList(result.connected_component_sizes_t2),
    _cell(result.connected_candidates_generated),
    _cell(result.connected_candidates_valid),
    _cell(result.connected_candidates_scored),
    _cell(result.connected_candidates_ortools),
    _cell(result.connected_candidates_vroom),
    _cell(result.connected_cut_edges),
    _cell(result.connected_cut_length_m),
    _cell(result.connected_cross_neighbors),
    _cell(result.connected_enclave_points),
    _cell(result.connected_selected_seed),
    _cell(result.connected_fallback_used),
    _cell(result.connected_error),
    _cell(result.connected_vroom_calls),
    _cell(result.selected_sequencer),
    _cell(result.final_selection_reason),
    _cell(result.ortools_total_duration_s),
    _cell(result.ortools_total_distance_m),
    _cell(result.vroom_total_duration_s),
    _cell(result.vroom_total_distance_m),
    _cell(result.partition_solver),

    // --- recentrage ORS-first ---
    _cell(result.connected_ors_reference_available),
    _cell(result.connected_ors_reference_duration_s),
    _cell(result.connected_ors_reference_distance_m),
    _cellList(result.connected_ors_reference_sizes),
    _cell(result.connected_ors_reference_components_t1),
    _cell(result.connected_ors_reference_components_t2),
    _cell(result.connected_ors_reference_time_limit_hit),
    _cell(result.connected_ors_reference_fallback_used),
    _cell(result.connected_ors_reference_solver_status),
    _cell(result.connected_ors_reference_solve_ms),
    _cell(result.connectivity_penalty_duration_s),
    _cell(result.connectivity_penalty_distance_m),
    _cell(result.connectivity_penalty_reliable),
    _cell(result.connectivity_penalty_note),
    _cell(result.connected_stage_timings_text),
    _cell(result.connected_stage_budget_exhausted),
    _cell(result.connected_generation_expired_after),
    _cell(result.connected_ors_repair_candidates_raw),
    _cell(result.connected_ors_repair_candidates_unique),
    _cell(result.connected_ors_repair_reached_ortools),
    _cell(result.connected_ors_repair_best_proxy_rank),
    _cell(result.connected_ors_repair_is_winner),
    _cell(result.connected_prescore_refined),
    _cell(result.connected_prescore_budget_exhausted),
    _cell(result.connected_winner_proxy_rank_rough),
    _cell(result.connected_winner_proxy_rank_refined),
    _cell(result.connected_winner_ortools_rank),
    _cell(result.connected_matrix_hash),
    _cell(result.connected_per_source_text),
    _cell(result.post_optimization_kept),
    _cell(result.post_optimization_note),

    // --- stratégie expérimentale VROOM local + ALNS territoriale ---
    // hyb est {} pour les quatre stratégies de production : toutes ces
    // cellules restent alors vides, aucune ligne existante ne change de forme.
    _cell(hyb.hybrid_error),
    _cell(hyb.local_vroom_enabled),
    _cell(hyb.local_vroom_version),
    _cell(hyb.common_rescore_duration_s),
    _cell(hyb.common_rescore_distance_m),
    _cell(hyb.common_rescore_matrix_hash),
    _cell(hyb.joint_direct_valid),
    _cell(hyb.joint_direct_duration_s),
    _cellList(hyb.joint_direct_sizes),
    _cell(hyb.joint_nucleus_attempted),
    _cell(hyb.joint_nucleus_valid),
    _cell(hyb.joint_nucleus_best_duration_s),
    _cell(hyb.route_first_unique),
    _cell(hyb.route_first_best_duration_s),
    _cell(hyb.joint_alns_iterations),
    _cell(hyb.joint_alns_accepted),
    _cell(hyb.joint_alns_seed),
    _cell(hyb.joint_alns_best_duration_s),
    _cell(hyb.joint_finalists),
    _cell(hyb.joint_finalists_local_vroom_solved),
    _cell(hyb.joint_finalists_reused),
    _cell(hyb.joint_solutions_considered),
    _cell(hyb.joint_selected_source),
    _cell(hyb.joint_selected_duration_s),
    _cell(hyb.joint_selected_distance_m),
    _cellList(hyb.joint_selected_sizes),
    _cellList(hyb.joint_selected_components),
    _cell(hyb.joint_selected_enclaves),
    _cell(hyb.joint_territorial_level),
    _cell(hyb.joint_territorial_max_enclaves),
    _cell(hyb.joint_territorial_admissible),
    _cell(hyb.joint_territorial_fallback_used),
    _cell(hyb.joint_territorial_fallback_reason),
    _cellList(hyb.joint_territorial_thresholds),
    _cellCounts(hyb.joint_territorial_level_counts),
    // Le plafond configuré, à lire en regard de local_vroom_attempted :
    // "4 lancées sur 4 autorisées" ne se voit pas sans lui.
    _cell(hyb.local_vroom_max_solves),
    _cell(hyb.local_vroom_attempted),
    _cell(hyb.local_vroom_succeeded),
    _cell(hyb.local_vroom_failed),
    _cell(hyb.local_vroom_timed_out),
    _cell(hyb.local_vroom_reused),
    _cell(hyb.local_vroom_skipped_for_time),
    _cell(hyb.local_vroom_elapsed_ms),
    _cell(hyb.local_vroom_stop_reason),
    _cell(hyb.local_vroom_last_error),
    _cell(_stageMs(stages, "matrix")),
    _cell(_stageMs(stages, "joint_direct")),
    _cell(_stageMs(stages, "route_first")),
    _cell(_stageMs(stages, "joint_nucleus")),
    _cell(_stageMs(stages, "joint_alns")),
    _cell(_stageMs(stages, "alns_refine")),
    _cell(_stageMs(stages, "joint_finalists")),
    // Texte de repli : si le backend ajoute un bloc, son temps apparaît ici
    // au lieu d'être perdu en silence.
    _stagesText(stages, "elapsed_ms"),
    _stagesText(stages, "stop_reason"),
    _cell(hyb.total_elapsed_ms),
    _cell(hyb.soft_limit_reached)
  ];

  sheet.appendRow(row);
}


/**
 * Bloc de diagnostic de la stratégie expérimentale.
 *
 * Il voyage dans result.ors_matrix.hybrid, là où la stratégie connexe passe
 * par result.ors_matrix.connected. Rend {} pour les quatre stratégies de
 * production, qui ne produisent pas ce bloc : les colonnes restent vides
 * plutôt que de lever.
 */
function _hybridDiag(result) {
  const meta = (result && result.ors_matrix) || {};
  return meta.hybrid || {};
}


/** Temps consommé par un bloc, "" si le bloc n'a pas tourné. */
function _stageMs(stages, name) {
  for (var i = 0; i < stages.length; i++) {
    if (stages[i] && stages[i].stage === name) return stages[i].elapsed_ms;
  }
  return "";
}


/** "bloc=valeur;bloc=valeur" sur tous les blocs réellement renvoyés. */
function _stagesText(stages, field) {
  const parts = [];
  for (var i = 0; i < stages.length; i++) {
    const stage = stages[i];
    if (!stage || !stage.stage) continue;
    parts.push(stage.stage + "=" + _cell(stage[field]));
  }
  return parts.join(";");
}


/** Sérialise un dictionnaire {clé: nombre} en "clé=valeur;clé=valeur". */
function _cellCounts(v) {
  if (v === null || v === undefined) return "";
  const keys = Object.keys(v).sort();
  const parts = [];
  for (var i = 0; i < keys.length; i++) {
    parts.push(keys[i] + "=" + v[keys[i]]);
  }
  return parts.join(";");
}


/** Rend "" pour null/undefined, la valeur sinon. Ne lève jamais. */
function _cell(v) {
  return (v === null || v === undefined) ? "" : v;
}


/** Sérialise une liste en texte lisible dans une cellule. "" si absente. */
function _cellList(v) {
  if (v === null || v === undefined) return "";
  return Array.isArray(v) ? v.join(" / ") : String(v);
}


// =========================
// CARTE DES DEUX TOURNÉES
// =========================
// Nom EXACT du fichier HTML à créer dans l'éditeur Apps Script
// (Fichier > Nouveau > Fichier HTML), sans l'extension.
const MAP_HTML_FILE  = "CarteTournees";

// Le payload d'un run de 62 points pèse ~8 Ko. PropertiesService plafonne à
// 9 Ko par valeur : trop juste dès que les adresses s'allongent. Une cellule
// de feuille accepte 50 000 caractères, avec en prime la persistance entre
// sessions. D'où cette feuille masquée à une seule cellule.
const MAP_DATA_SHEET = "_CarteData";

const MAP_COLORS = ["#1f5fa9", "#d35400"];   // Tournée 1 bleu, Tournée 2 orange foncé


// Identifiant du classeur, mémorisé depuis le classeur lui-même.
// SpreadsheetApp.getActive() ne veut rien dire dans une Web App : il n'y a
// ni classeur actif, ni interface. Sans cet identifiant explicite, doGet
// n'aurait aucun moyen fiable de retrouver les données.
const PROP_SPREADSHEET_ID = "TOURNEES_SPREADSHEET_ID";


/** Enregistre l'ID du classeur. Appelé depuis le classeur, jamais ailleurs. */
function _memoriserClasseur_() {
  try {
    PropertiesService.getScriptProperties()
      .setProperty(PROP_SPREADSHEET_ID, SpreadsheetApp.getActive().getId());
  } catch (e) {
    // Contexte sans classeur actif : rien à mémoriser, rien à signaler.
  }
}


/**
 * Le classeur, dans les deux contextes.
 *
 * Depuis le classeur, getActive() suffit. Depuis la Web App il rend null :
 * on ouvre alors explicitement par identifiant. openById s'exécute avec les
 * droits de l'utilisateur accédant et LÈVE s'il n'a pas accès — c'est
 * exactement la barrière voulue, et elle est portée par Google, pas par nous.
 */
function _classeur_() {
  const actif = SpreadsheetApp.getActive();
  if (actif) return actif;
  const id = PropertiesService.getScriptProperties()
    .getProperty(PROP_SPREADSHEET_ID);
  if (!id) throw new Error("Classeur non mémorisé.");
  return SpreadsheetApp.openById(id);
}


function _mapDataSheet_(createIfMissing) {
  const ss = _classeur_();
  let sh = ss.getSheetByName(MAP_DATA_SHEET);
  if (!sh && createIfMissing) {
    sh = ss.insertSheet(MAP_DATA_SHEET);
    sh.getRange(1, 2).setValue(
      "Stockage technique de la dernière carte. Ne pas modifier.");
    sh.hideSheet();
  }
  return sh;
}


function _saveCartePayload_(payload) {
  _mapDataSheet_(true).getRange(1, 1).setValue(JSON.stringify(payload));
}


/**
 * Appelée depuis le HTML par google.script.run. Doit rester au niveau global.
 * Retourne le JSON du dernier run, ou null s'il n'y en a pas — le HTML affiche
 * alors « Aucune tournée récente disponible ».
 */
function getCarteTourneesPayload() {
  try {
    const sh = _mapDataSheet_(false);
    if (!sh) return null;
    const v = sh.getRange(1, 1).getValue();
    return v ? String(v) : null;
  } catch (e) {
    // Utilisateur sans accès au classeur : aucune donnée ne sort d'ici.
    return null;
  }
}


// =========================
// WEB APP
// =========================
// Une seule page responsive, la même sur ordinateur, Android et iPhone.
//
// doGet n'utilise NI SpreadsheetApp.getActive(), NI getUi(), NI aucune notion
// de feuille active : rien de tout cela n'existe hors du classeur. Le
// classeur est retrouvé par son identifiant mémorisé, puis ouvert avec les
// droits de l'utilisateur qui consulte la page.

/** Page de message. Ne contient jamais la moindre donnée de tournée. */
function _pageMessage_(titre, message) {
  const esc = function (s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;");
  };
  return HtmlService.createHtmlOutput(
      '<div style="font-family:Arial,Helvetica,sans-serif;max-width:520px;'
    + 'margin:12vh auto;padding:0 24px;line-height:1.6;color:#222">'
    + "<h1 style=\"font-size:19px\">" + esc(titre) + "</h1>"
    + "<p>" + esc(message) + "</p></div>")
    .setTitle("Carte des tournées")
    .addMetaTag("viewport", "width=device-width, initial-scale=1");
}


/**
 * Point d'entrée de la Web App : un instantané partagé, et RIEN d'autre.
 *
 * Le jeton est la seule clé. Sans jeton valide, aucune donnée ne sort d'ici :
 * la Web App ne sert plus « la dernière carte », usage revenu au dialogue du
 * classeur. C'est ce qui permet d'ouvrir le déploiement au partage sans
 * ouvrir en même temps l'accès au dernier run à quiconque connaît l'adresse.
 *
 * La page servie est le fichier d'archive lui-même, relu depuis Drive : le
 * lien et le fichier téléchargé montrent donc exactement la même carte.
 */
function doGet(e) {
  const jeton = (e && e.parameter && e.parameter[SHARE_PARAM])
    ? String(e.parameter[SHARE_PARAM]) : "";

  if (!jeton) {
    return _pageMessage_("Lien incomplet",
      "Ce lien ne désigne aucune carte. Demandez à la personne qui vous l'a "
      + "envoyé de vous transmettre le lien en entier.");
  }

  var partage = null;
  try {
    partage = _lirePartage_(jeton);
  } catch (err) {
    partage = null;
  }
  if (!partage) {
    return _pageMessage_("Carte indisponible", MSG_PARTAGE_INDISPONIBLE);
  }

  var html = "";
  try {
    html = DriveApp.getFileById(partage.fileId)
      .getBlob().getDataAsString("UTF-8");
  } catch (err) {
    // Fichier supprimé de Drive : le partage n'a plus d'objet, et la réponse
    // reste la même que pour un jeton inconnu.
    return _pageMessage_("Carte indisponible", MSG_PARTAGE_INDISPONIBLE);
  }

  return HtmlService.createHtmlOutput(html)
    .setTitle("Carte des tournées")
    .addMetaTag("viewport",
                "width=device-width, initial-scale=1, viewport-fit=cover");
}


// =========================
// URL DE LA WEB APP : SOURCE UNIQUE
// =========================
// Toute ouverture « en grand » — entrée de menu comme bouton de la carte —
// passe par _getWebAppUrl_ et par rien d'autre.
//
// Pourquoi une validation, alors que la plateforme est censée rendre la
// bonne valeur : parce que l'ancienne version rendait TELLE QUELLE la chaîne
// reçue, sans jamais la regarder. Une adresse de brouillon en /dev, une
// adresse de classeur, ou une adresse Drive héritée du chemin d'export
// partaient donc directement dans window.open. Drive n'affiche pas les
// fichiers HTML : il répond « Impossible d'ouvrir le fichier pour le
// moment », et l'utilisateur n'avait aucun moyen de savoir d'où venait
// l'adresse.
//
// L'archive Drive naît dans _deposerCarteSurDrive_ et n'en sort JAMAIS pour
// alimenter une ouverture de carte. Les deux mondes ne se croisent plus :
// l'un sert à consulter, l'autre à archiver.

const WEB_APP_URL_SUFFIXE = "/exec";

// Fragments qui disqualifient une adresse. Ce sont ceux d'un fichier Drive,
// d'un export, d'un téléchargement ou d'un aperçu — jamais ceux d'une Web App.
const WEB_APP_URL_INTERDITS = [
  "drive.google.com", "/file/d/", "uc?export=", "/view", "/download"
];

const MSG_WEB_APP_INDISPONIBLE =
    "La Web App n'est pas encore disponible. Mettez à jour ou déployez "
  + "l'application Web Apps Script.";


/**
 * Rend l'adresse si elle est utilisable pour ouvrir la carte, "" sinon.
 *
 * Une adresse refusée n'est jamais remplacée par un repli : mieux vaut un
 * message clair qu'un lien qui mène ailleurs que là où il annonce.
 */
function _validerUrlWebApp_(brut) {
  const url = String(brut === null || brut === undefined ? "" : brut).trim();
  if (!url) return "";
  if (url.indexOf("https://") !== 0) return "";

  const bas = url.toLowerCase();
  for (var i = 0; i < WEB_APP_URL_INTERDITS.length; i++) {
    if (bas.indexOf(WEB_APP_URL_INTERDITS[i]) !== -1) return "";
  }

  // Un déploiement d'application Web se termine par /exec. /dev est l'adresse
  // de brouillon : réservée aux éditeurs du script, elle n'ouvrirait rien
  // pour la personne qui conduit la tournée.
  const fin = bas.length - WEB_APP_URL_SUFFIXE.length;
  if (fin < 0 || bas.lastIndexOf(WEB_APP_URL_SUFFIXE) !== fin) return "";

  return url;
}


/**
 * Source de vérité unique de l'URL Web App : ScriptApp.getService().getUrl().
 *
 * Ne lève jamais — avant le premier déploiement, ScriptApp n'a pas d'URL à
 * donner, et l'appelant doit alors le dire plutôt qu'afficher un lien mort.
 */
function _getWebAppUrl_() {
  try {
    return _validerUrlWebApp_(ScriptApp.getService().getUrl());
  } catch (e) {
    return "";
  }
}


/**
 * Surface appelée depuis le HTML par google.script.run : doit rester au
 * niveau global. Aucune logique ici, uniquement la délégation — le helper
 * reste le seul endroit où une adresse d'ouverture est produite et validée.
 */
function getWebAppUrl() {
  return _getWebAppUrl_();
}


/**
 * Entrée de menu « Ouvrir la carte » : rouvre la DERNIÈRE carte produite.
 *
 * Exactement la même fenêtre que celle qui s'ouvre en fin d'optimisation, et
 * exactement le même payload : rien n'est recalculé, aucune optimisation
 * n'est relancée, aucun onglet n'est ouvert, aucune adresse n'est construite.
 *
 * La Web App n'intervient plus ici. Elle ne sert qu'au partage d'un
 * instantané, ce qui est un autre besoin et un autre chemin.
 */
function ouvrirLaCarte() {
  if (!getCarteTourneesPayload()) {
    SpreadsheetApp.getActive().toast(MSG_AUCUNE_CARTE, "Carte", 6);
    return;
  }
  _afficherDialogueCarte_();
}


/**
 * Construit le payload de la carte à partir du run qui vient d'aboutir.
 *
 * result.tournee_N contient les identifiants DANS L'ORDRE DE PASSAGE, dépôt
 * inclus aux deux extrémités : le backend renvoie [start] + points + [end].
 * D'où role = "start" au premier rang, "end" au dernier, "collection" entre.
 *
 * Les latitudes, longitudes et libellés viennent du snapshot du Sheet passé en
 * argument — aucun appel réseau, aucun recalcul, aucune réoptimisation.
 */
function buildCartePayload(result, params, points) {

  const latMap = {}, lonMap = {}, addrMap = {};
  for (let i = 0; i < points.length; i++) {
    const sid = String(points[i].id);
    latMap[sid]  = points[i].lat;
    lonMap[sid]  = points[i].lon;
    addrMap[sid] = points[i].address || "";
  }

  const specs = [
    { key: "tournee_1", label: "Tournée 1" },
    { key: "tournee_2", label: "Tournée 2" }
  ];

  const routes = [];
  const skipped = [];

  for (let s = 0; s < specs.length; s++) {
    const ids = result[specs[s].key] || [];
    const pts = [];
    let order = 0;

    for (let k = 0; k < ids.length; k++) {
      const sid = String(ids[k]);
      const lat = Number(latMap[sid]);
      const lon = Number(lonMap[sid]);

      let role = "collection";
      if (k === 0) role = "start";
      else if (k === ids.length - 1) role = "end";

      // Coordonnée absente ou illisible : le point est écarté avec son
      // identifiant, sans empêcher l'affichage du reste de la carte.
      if (!isFinite(lat) || !isFinite(lon) || latMap[sid] === "" || lonMap[sid] === "") {
        skipped.push(specs[s].label + " / " + sid);
        continue;
      }

      if (role === "collection") order++;

      pts.push({
        order: (role === "start") ? "D" : (role === "end") ? "A" : order,
        id: sid,
        label: addrMap[sid],
        lat: lat,
        lon: lon,
        role: role
      });
    }

    routes.push({
      label: specs[s].label,
      color: MAP_COLORS[s % MAP_COLORS.length],
      distance_km:  _cell(result[specs[s].key + "_km"]),
      duration_min: _cell(result[specs[s].key + "_min"]),
      points: pts
    });
  }

  return {
    generated_at: new Date().toISOString(),
    strategy: result.strategy_used || result.strategy_requested || params.strategy || "",
    points_signature: result.points_signature || "",
    skipped: skipped,
    routes: routes,

    // Certificat territorial, affiché tel quel par la carte. Absent des
    // stratégies qui ne le produisent pas : la carte masque alors le bloc.
    territorial: {
      partition: result.territorial_partition,
      method: result.territorial_method,
      locked: result.territorial_membership_locked,
      violations: result.territorial_side_violations,
      angle_deg: result.territorial_separator_angle_deg,
      margin_m: result.territorial_separator_margin_m,
      status: result.territorial_overlap_status,
      error: result.territorial_error
    },

    // Certificat de connexite, affiché par la carte pour le mode connexe.
    connected: {
      partition: result.connected_partition,
      method: result.connected_method,
      locked: result.connected_membership_locked,
      sizes: result.connected_target_sizes,
      components_t1: result.connected_components_t1,
      components_t2: result.connected_components_t2,
      component_sizes_t1: result.connected_component_sizes_t1,
      component_sizes_t2: result.connected_component_sizes_t2,
      cut_edges: result.connected_cut_edges,
      cut_length_m: result.connected_cut_length_m,
      enclave_points: result.connected_enclave_points,
      sequencer: result.selected_sequencer,
      error: result.connected_error
    }
  };
}


// Titre et dimensions du dialogue. Définis une seule fois : l'ouverture
// automatique en fin d'optimisation et l'entrée de menu doivent produire
// exactement la même fenêtre, pas deux fenêtres qui se ressemblent.
const MAP_DIALOG_TITLE  = "Carte des deux tournées";
const MAP_DIALOG_WIDTH  = 1200;
const MAP_DIALOG_HEIGHT = 800;

const MSG_AUCUNE_CARTE =
  "Aucune carte disponible. Lancez d'abord une optimisation.";


/**
 * UNIQUE chemin d'affichage de la carte.
 *
 * Le gabarit part sans données : il réclame ensuite le dernier payload par
 * google.script.run.getCarteTourneesPayload(). Les deux parcours — fin
 * d'optimisation et entrée de menu — convergent ici, et nulle part ailleurs.
 * Aucun onglet, aucune Web App, aucune URL : un dialogue, point.
 */
function _afficherDialogueCarte_() {
  const out = HtmlService.createHtmlOutputFromFile(MAP_HTML_FILE)
    .setWidth(MAP_DIALOG_WIDTH)
    .setHeight(MAP_DIALOG_HEIGHT);
  SpreadsheetApp.getUi().showModalDialog(out, MAP_DIALOG_TITLE);
}


/** Construit, enregistre puis affiche la carte du run courant. */
function afficherCarteTournees(result, params, points) {
  _saveCartePayload_(buildCartePayload(result, params, points));
  _afficherDialogueCarte_();
}


// =========================
// GÉOMÉTRIE ROUTIÈRE DE LA CARTE
// =========================
// Entièrement séparé de l'optimisation. Ces appels partent APRÈS qu'un run
// est terminé et mesuré : ils n'entrent ni dans le temps Benchmark, ni dans
// les appels comptés de l'optimisation, et un échec ne peut pas transformer
// une optimisation réussie en erreur.
//
// La clé ORS n'apparaît nulle part ici : c'est le backend qui la détient,
// et c'est la seule raison pour laquelle cet aller-retour existe.

const MAP_GEOMETRY_TIMEOUT_MS = 25000;


/** Coordonnées [lon, lat] des deux tournées, dans l'ordre EXACT de passage. */
function _coordsDesRoutes_(payload) {
  const routes = (payload && payload.routes) || [];
  const out = [];
  for (var r = 0; r < routes.length; r++) {
    const pts = routes[r].points || [];
    const coords = [];
    for (var i = 0; i < pts.length; i++) {
      const lat = Number(pts[i].lat), lon = Number(pts[i].lon);
      if (isFinite(lat) && isFinite(lon)) coords.push([lon, lat]);
    }
    out.push(coords);
  }
  return out;
}


/**
 * Demande au backend le tracé routier des deux tournées.
 *
 * Appelée depuis le HTML par google.script.run : doit rester au niveau
 * global. Ne lève jamais — la carte doit pouvoir garder ses segments
 * indicatifs sans qu'aucune erreur ne remonte à l'utilisateur.
 *
 * @param {string} coordsJson  [[[lon,lat],...],[[lon,lat],...]]
 * @return {string} JSON de la réponse backend, ou d'un repli local.
 */
function getCarteGeometrie(coordsJson) {
  try {
    const routes = JSON.parse(coordsJson);
    if (!Array.isArray(routes) || routes.length !== 2) {
      return JSON.stringify({geometries: null, status: "invalid_request",
                             cache_hit: false, calls: 0, elapsed_ms: 0,
                             fallback_used: true});
    }

    // Aucun réveil du serveur ici, contrairement à callAPI : la carte est
    // déjà affichée et ne doit pas attendre un démarrage à froid de Render
    // pendant une minute. Si le serveur dort, on retombe sur les pointillés.
    const response = UrlFetchApp.fetch(API_BASE + "/map-geometry", {
      method: "post",
      contentType: "application/json",
      payload: JSON.stringify({routes: routes, profile: "driving-car"}),
      muteHttpExceptions: true
    });

    const code = response.getResponseCode();
    const text = response.getContentText();
    if (code !== 200 || text.charAt(0) !== "{") {
      return JSON.stringify({geometries: null, status: "http_" + code,
                             cache_hit: false, calls: 0, elapsed_ms: 0,
                             fallback_used: true});
    }
    return text;

  } catch (e) {
    return JSON.stringify({geometries: null, status: "unreachable",
                           cache_hit: false, calls: 0, elapsed_ms: 0,
                           fallback_used: true,
                           error: String(e && e.message ? e.message : e)});
  }
}


/**
 * Recopie du payload avec les géométries greffées sur les tournées.
 *
 * La géométrie n'est JAMAIS réécrite dans _CarteData : une cellule de
 * feuille plafonne à 50 000 caractères, et deux tracés routiers la feraient
 * sauter. Elle ne vit que dans la fenêtre et dans le fichier exporté.
 */
function _payloadAvecGeometries_(json, geometries) {
  if (!geometries) return json;
  const payload = JSON.parse(json);
  const routes = payload.routes || [];
  for (var i = 0; i < routes.length && i < geometries.length; i++) {
    if (Array.isArray(geometries[i]) && geometries[i].length > 1) {
      routes[i].geometry = geometries[i];
    }
  }
  return JSON.stringify(payload);
}


/** Nom de fichier déterministe, réduit à une liste de caractères sûrs. */
function _nomFichierCarte_(signature) {
  const propre = String(signature || "sans-signature")
    .replace(/[^A-Za-z0-9_-]/g, "")
    .slice(0, 40) || "sans-signature";
  const stamp = Utilities.formatDate(
    new Date(), Session.getScriptTimeZone(), "yyyy-MM-dd_HH-mm");
  return "carte_tournees_" + propre + "_" + stamp + ".html";
}


/** Crée le fichier Drive et rend de quoi l'ouvrir. Aucun secret n'y entre. */
function _deposerCarteSurDrive_(html, signature) {
  const name = _nomFichierCarte_(signature);
  const file = DriveApp.createFile(name, html, MimeType.HTML);
  return {
    id: file.getId(),
    name: name,
    sizeKb: Math.round(html.length / 1024),
    url: file.getUrl(),
    downloadUrl: "https://drive.google.com/uc?export=download&id=" + file.getId()
  };
}


// =========================
// PARTAGE D'UN INSTANTANÉ DE CARTE
// =========================
// Pourquoi ce détour plutôt que d'envoyer le fichier : Drive ne PRÉVISUALISE
// pas le HTML qu'il n'a pas produit. Il propose de le télécharger, et sur
// téléphone cela ne mène à rien d'utilisable — d'où « Impossible d'ouvrir le
// fichier pour le moment ». Un fichier reçu par messagerie ne s'ouvre pas
// mieux : l'aperçu d'iOS n'exécute pas JavaScript, et Chrome Android refuse
// souvent d'ouvrir un HTML local.
//
// Ce qui marche partout, sans installation ni compte, c'est une vraie adresse
// HTTPS. La Web App en est déjà une. Elle sert donc l'instantané — le MÊME
// fichier que l'archive, relu depuis Drive — identifié par un jeton.
//
// Le jeton est la seule clé. Sans lui la Web App ne rend rien : elle n'expose
// plus « la dernière carte », cet usage étant revenu au dialogue.

const SHARE_SHEET = "_CartesPartagees";
const SHARE_HEADERS = ["Jeton", "Fichier Drive", "Nom", "Créée le",
                       "Expire le", "Signature jeu"];

// Paramètre d'URL portant le jeton. Court, parce qu'il voyage dans un lien
// que quelqu'un va recopier ou coller dans une messagerie.
const SHARE_PARAM = "c";

// Au-delà, le lien cesse de fonctionner sans qu'on ait à y penser. Une carte
// de tournée n'a de sens que peu de temps.
const SHARE_TTL_DAYS = 30;

const MSG_PARTAGE_INDISPONIBLE =
    "Ce lien n'est plus valable. Il a peut-être expiré, ou le partage a été "
  + "révoqué.";


/**
 * Jeton de partage : deux UUID concaténés, tirets retirés.
 *
 * 64 caractères hexadécimaux, soit largement au-delà de ce qu'une recherche
 * exhaustive peut atteindre. C'est ce qui rend le lien non devinable — et
 * c'est aussi pourquoi il ne doit être transmis qu'aux personnes concernées.
 */
function _jetonPartage_() {
  return (Utilities.getUuid() + Utilities.getUuid()).replace(/-/g, "");
}


/** Registre des partages. Feuille masquée, lisible et modifiable à la main. */
function _sharesSheet_(createIfMissing) {
  const ss = _classeur_();
  let sh = ss.getSheetByName(SHARE_SHEET);
  if (!sh && createIfMissing) {
    sh = ss.insertSheet(SHARE_SHEET);
    sh.getRange(1, 1, 1, SHARE_HEADERS.length).setValues([SHARE_HEADERS]);
    sh.getRange(1, 1, 1, SHARE_HEADERS.length)
      .setBackground("#434343").setFontColor("#ffffff").setFontWeight("bold");
    sh.setFrozenRows(1);
    sh.hideSheet();
  }
  return sh;
}


/**
 * Enregistre un instantané partageable et rend son jeton.
 *
 * Seuls un identifiant de fichier et des dates entrent ici : ni adresse, ni
 * coordonnée, ni géométrie. Les données de la carte restent dans le fichier
 * Drive, dont ce registre ne fait que désigner l'emplacement.
 */
function _enregistrerPartage_(fileId, nom, signature) {
  const jeton = _jetonPartage_();
  const cree = new Date();
  const expire = new Date(cree.getTime() + SHARE_TTL_DAYS * 24 * 3600 * 1000);
  _sharesSheet_(true).appendRow(
    [jeton, String(fileId), String(nom || ""), cree, expire,
     String(signature || "")]);
  return jeton;
}


/**
 * Retrouve un partage valide, ou null.
 *
 * Un jeton inconnu, expiré ou effacé rend null, sans distinction : la page
 * d'erreur ne doit pas révéler qu'un jeton a existé.
 */
function _lirePartage_(jeton) {
  const cle = String(jeton || "").trim();
  if (!cle) return null;

  const sh = _sharesSheet_(false);
  if (!sh) return null;

  const lastRow = sh.getLastRow();
  if (lastRow < 2) return null;

  const rows = sh.getRange(2, 1, lastRow - 1, SHARE_HEADERS.length).getValues();
  const maintenant = new Date();
  for (var i = 0; i < rows.length; i++) {
    if (String(rows[i][0]).trim() !== cle) continue;
    const expire = rows[i][4];
    if (expire instanceof Date && expire.getTime() < maintenant.getTime()) {
      return null;
    }
    const fileId = String(rows[i][1] || "").trim();
    return fileId ? {fileId: fileId, nom: String(rows[i][2] || "")} : null;
  }
  return null;
}


/**
 * Révoque TOUS les partages : les liens déjà envoyés cessent de fonctionner.
 *
 * Sans entrée de menu, volontairement — c'est une action rare, et le menu
 * doit rester celui de la conduite des tournées. Elle s'exécute depuis
 * l'éditeur Apps Script. Révoquer un seul lien se fait en supprimant sa
 * ligne dans la feuille « _CartesPartagees », qu'il suffit d'afficher.
 *
 * Les fichiers Drive ne sont pas supprimés : ils restent votre archive.
 */
function revoquerPartagesCarte() {
  const sh = _sharesSheet_(false);
  if (!sh || sh.getLastRow() < 2) return 0;
  const n = sh.getLastRow() - 1;
  sh.deleteRows(2, n);
  return n;
}


/**
 * Export déclenché par le bouton de la fenêtre carte.
 *
 * Les géométries déjà chargées dans la fenêtre sont transmises telles
 * quelles : aucun nouvel appel Directions n'est émis, l'export ne coûte donc
 * rien de plus que le tracé déjà affiché. Sans géométrie, l'export part en
 * segments indicatifs plutôt que d'attendre.
 *
 * Ne relance aucune optimisation, n'écrit ni dans le Benchmark ni dans
 * _CarteData.
 */
function exporterCarteDepuisDialogue(geometriesJson) {
  const json = getCarteTourneesPayload();
  if (!json) return JSON.stringify({error: "Aucune carte enregistrée."});

  var geometries = null;
  try {
    geometries = geometriesJson ? JSON.parse(geometriesJson) : null;
  } catch (e) {
    geometries = null;
  }

  const enrichi = _payloadAvecGeometries_(json, geometries);
  const html = _buildStandaloneCarteHtml_(enrichi);

  var signature = "";
  try { signature = JSON.parse(json).points_signature || ""; } catch (e) {}

  const info = _deposerCarteSurDrive_(html, signature);
  info.withGeometry = !!(geometries && geometries.length);

  // Lien de consultation. Un échec ici ne fait pas échouer l'export : le
  // fichier existe déjà, et la fenêtre le dira plutôt que de tout perdre.
  try {
    const base = _getWebAppUrl_();
    if (base) {
      info.shareUrl = base + "?" + SHARE_PARAM + "="
        + _enregistrerPartage_(info.id, info.name, signature);
    } else {
      info.shareError = MSG_WEB_APP_INDISPONIBLE;
    }
  } catch (e) {
    info.shareError = "Lien de partage indisponible : "
      + (e && e.message ? e.message : e);
  }

  return JSON.stringify(info);
}


/**
 * Fabrique une carte AUTONOME : le gabarit HTML avec les données du run
 * injectées dedans. Le fichier obtenu s'ouvre n'importe où — poste local,
 * pièce jointe, Drive — sans Apps Script et sans accès au classeur.
 *
 * L'amorçage du gabarit teste window.TOURNEES_PAYLOAD en premier : il rend
 * donc directement la carte et n'appelle jamais google.script.run.
 */
function _buildStandaloneCarteHtml_(payloadJson) {

  const tpl = HtmlService.createHtmlOutputFromFile(MAP_HTML_FILE).getContent();

  if (tpl.indexOf("</head>") === -1) {
    throw new Error("Gabarit " + MAP_HTML_FILE + " inattendu : balise </head> absente.");
  }

  // Les libellés viennent du Sheet et peuvent contenir n'importe quoi.
  // Échapper "<" neutralise </script> et <!-- , qui sortiraient du bloc.
  // U+2028 et U+2029 sont des terminateurs de ligne en JavaScript. Ils
  // doivent donc etre ecrits en forme echappee dans la regex : places tels
  // quels, ils cassent le litteral et le fichier ne se parse plus. Et jamais
  // l'espace ordinaire U+0020, qui n'a rien a neutraliser ici.
  const safe = payloadJson
    .replace(/</g, "\\u003c")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");

  return tpl.replace("</head>",
    "<script>window.TOURNEES_PAYLOAD = " + safe + ";</script>\n</head>");
}


/**
 * Exporte la dernière carte en un fichier HTML autonome déposé sur Drive,
 * puis affiche son lien.
 *
 * Retirée du menu : l'export existe déjà dans la carte, via le bouton
 * « Exporter », qui réutilise les géométries déjà chargées et ne coûte donc
 * aucun appel de plus. Cette variante reste exécutable depuis l'éditeur Apps
 * Script pour archiver une carte sans rouvrir la fenêtre.
 *
 * Le fichier est créé PRIVÉ. Le partage reste une décision explicite, à
 * prendre dans Drive : ces données sont opérationnelles.
 */
function exporterCartePartageable() {

  const json = getCarteTourneesPayload();
  if (!json) {
    SpreadsheetApp.getActive().toast(
      "Aucune carte enregistrée. Lancez une optimisation.", "Export", 5);
    return;
  }

  // Tracé routier demandé à la volée. Sur cache backend il ne coûte aucun
  // appel ; sinon deux au maximum. Un échec n'empêche pas l'export : le
  // fichier part alors avec ses segments indicatifs.
  var payload = null;
  try { payload = JSON.parse(json); } catch (e) {}
  var geometries = null;
  var geoStatus = "non_demande";
  if (payload) {
    try {
      const reponse = JSON.parse(
        getCarteGeometrie(JSON.stringify(_coordsDesRoutes_(payload))));
      geometries = reponse.geometries;
      geoStatus = reponse.status;
    } catch (e) {
      geoStatus = "erreur";
    }
  }

  const html = _buildStandaloneCarteHtml_(
    _payloadAvecGeometries_(json, geometries));
  const info = _deposerCarteSurDrive_(html,
    payload ? payload.points_signature : "");
  const name = info.name;

  const esc = function (s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  };

  const body =
      '<div style="font-family:Arial,sans-serif;font-size:13px;line-height:1.7;padding:14px">'
    + "<b>Carte exportée.</b><br>"
    + "Fichier : <code>" + esc(name) + "</code>"
    + " &mdash; " + info.sizeKb + " Ko<br>"
    + "Tracé : " + (geometries ? "itinéraires routiers réels"
                               : "segments indicatifs (" + esc(geoStatus) + ")")
    + "<br><br>"
    + '<a href="' + esc(info.url) + '" target="_blank">Ouvrir dans Drive</a>'
    + " &nbsp;|&nbsp; "
    + '<a href="' + esc(info.downloadUrl)
    + '" target="_blank">Télécharger le fichier</a>'
    + "<br><br>"
    + "<b>Pour partager</b> : dans Drive, clic droit sur le fichier &rsaquo; Partager.<br>"
    + "Il est créé <b>privé</b> : rien n'est publié sans votre décision.<br><br>"
    + "<b>Pour consulter la carte</b> sur ordinateur ou sur téléphone, "
    + "utilisez <i>Tournées &rsaquo; Ouvrir la carte</i> : Drive n'affiche "
    + "pas les fichiers HTML, il propose seulement de les télécharger.<br><br>"
    + "<small>Le fichier contient les données du run. Il charge Leaflet et les "
    + "tuiles OpenStreetMap depuis Internet : une connexion reste nécessaire "
    + "pour l'afficher.</small></div>";

  SpreadsheetApp.getUi().showModalDialog(
    HtmlService.createHtmlOutput(body).setWidth(560).setHeight(300),
    "Carte partageable");
}




// =========================
// LANCER L'OPTIMISATION
// =========================
/**
 * @param {string=} strategyOverride  Forcé par les entrées de menu.
 *                                    Absent -> stratégie lue dans Paramètres!B6.
 * Une seule stratégie par exécution : enchaîner les trois dépasserait
 * la limite de 6 minutes d'Apps Script.
 */
function runOptimisation(strategyOverride) {

  ensureStrategyCell();

  const params = getParams();
  if (typeof strategyOverride === "string" && strategyOverride) {
    params.strategy = strategyOverride;
  }

  const points = getPoints();

  if (points.length === 0) {
    SpreadsheetApp.getActive().toast("Aucun point sélectionné !", "Erreur", 3);
    return;
  }

  SpreadsheetApp.getActive().toast(
    "Optimisation en cours... (" + points.length + " points, stratégie " + params.strategy + ")",
    "Info", 10
  );

  const result = callAPI(points, params);

  if (result.error) {
    SpreadsheetApp.getActive().toast("Erreur API : " + result.error, "Erreur", 5);
    return;
  }

  writeResult(result, params, points);

  // Une réponse 200 peut masquer une dégradation backend : VROOM indisponible
  // => vroom_ok faux => swaps jamais exécutés. Sans cette remontée, la ligne
  // Benchmark paraît normale alors qu'elle ne mesure pas la même chose.
  var degraded = "";
  if (result.vroom_used === false) {
    degraded = "VROOM indisponible (" + (result.vroom_error || "raison inconnue")
             + ") : swaps non exécutés";
  } else if (result.swap_stop_reason === "vroom_error") {
    degraded = "swaps non exécutés (vroom_error)";
  }

  appendBenchmark(result, params, points, { run_error: degraded });

  var vroomInfo = result.vroom_used ? "Vroom direct" : "K-Means + Vroom (fallback)";
  var stratLabel = result.strategy_used || params.strategy;
  SpreadsheetApp.getActive().toast(
    "Terminé ! Stratégie : " + stratLabel + " | " + vroomInfo,
    "Succès", 10
  );

  // Carte en DERNIER : un dialogue modal bloque le script jusqu'à sa fermeture,
  // les résultats doivent donc déjà être écrits. On n'arrive ici qu'après un run
  // réussi : callAPI lève sur une erreur HTTP et result.error sort plus haut,
  // donc aucune carte n'est ouverte après un échec.
  try {
    afficherCarteTournees(result, params, points);
  } catch (e) {
    SpreadsheetApp.getActive().toast(
      "Carte non affichée : " + (e && e.message ? e.message : e), "Carte", 8);
  }
}


// Wrappers : Apps Script n'autorise pas d'argument depuis une entrée de menu.
function runKmeans()           { runOptimisation("kmeans"); }
function runOrtoolsHaversine() { runOptimisation("ortools_haversine"); }
function runOrtoolsOrsMatrix() { runOptimisation("ortools_ors_matrix"); }
function runOrtoolsOrsMatrixConnected() { runOptimisation("ortools_ors_matrix_connected"); }

// Stratégie expérimentale. L'identifiant vient de la constante, jamais d'une
// chaîne recopiée : le libellé de menu et le nom envoyé à l'API ne peuvent
// pas diverger.
function runHybridLocalVroomTerritorial() { runOptimisation(EXP_STRATEGY); }



// =========================
// MENU PERSONNALISÉ
// =========================
/**
 * Menu « Tournées ».
 *
 * Trois entrées au premier niveau, et ce sont les trois gestes d'une journée :
 * choisir les points, optimiser, regarder la carte. Tout le reste — le
 * benchmark et les méthodes de comparaison — descend sous « Outils dev »,
 * parce que rien de tout cela ne sert à conduire une tournée.
 *
 * « Exporter la carte » quitte le menu : l'export existe déjà DANS la carte,
 * et le proposer deux fois laissait croire que l'archive Drive était le moyen
 * normal de consulter la carte. Elle ne l'est pas — Drive n'affiche pas les
 * fichiers HTML.
 *
 * Rien n'est supprimé pour autant : les stratégies et fonctions retirées du
 * menu visible restent définies, exécutables depuis l'éditeur Apps Script, et
 * le backend les accepte toujours.
 */
function onOpen() {
  // Le classeur est mémorisé à chaque ouverture : c'est la seule occasion où
  // son identifiant est connu de façon fiable, et la Web App en dépend.
  _memoriserClasseur_();

  const ui = SpreadsheetApp.getUi();
  ui.createMenu(MENU_RACINE_LABEL)
    .addItem("Sélectionner les points par ID", "ouvrirSelectionParId")
    .addItem(MENU_OPTIMISER_LABEL, "runHybridLocalVroomTerritorial")
    .addItem("Ouvrir la carte", "ouvrirLaCarte")
    .addSeparator()
    .addSubMenu(
      ui.createMenu("Outils dev")
        .addItem("Ouvrir le benchmark", "ouvrirBenchmark")
        .addSubMenu(
          ui.createMenu("Méthodes de comparaison")
            .addItem("K-means — référence", "runKmeans")
            .addItem("ORS connecté — comparaison",
                     "runOrtoolsOrsMatrixConnected")
        )
    )
    .addToUi();
}


/**
 * Ouvre la feuille Benchmark, en la créant au besoin.
 *
 * Aucune logique dupliquée : ensureBenchmarkSheet() reste seule responsable
 * de la création de la feuille et de la migration de ses colonnes. Ce
 * wrapper n'ajoute que la navigation, qu'une entrée de menu ne sait pas
 * faire elle-même.
 */
function ouvrirBenchmark() {
  const sheet = ensureBenchmarkSheet();
  SpreadsheetApp.getActive().setActiveSheet(sheet);
}


// =========================
// EFFACER TOURNÉES
// =========================
// N'efface que "Résultats". La feuille "Benchmark" est volontairement épargnée :
// c'est l'historique de comparaison des stratégies.
function clearResults() {
  const sheet = SpreadsheetApp.getActive().getSheetByName("Résultats");
  if (sheet) {
    sheet.clear();
    SpreadsheetApp.getActive().toast("Tournées effacées !", "Info", 3);
  }
}


// =========================
// RÉINITIALISER SÉLECTION
// =========================
function resetSelection() {
  const sheet = SpreadsheetApp.getActive().getSheetByName("Horodateurs");
  if (!sheet) return;

  const lastRow = sheet.getLastRow();
  if (lastRow < 2) return;

  sheet.getRange(2, 5, lastRow - 1, 1).uncheck();
  SpreadsheetApp.getActive().toast("Sélection réinitialisée !", "Info", 3);
}