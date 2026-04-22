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
    ["Point d'arrivée (ID)", ""]
  ];

  paramSheet.getRange(1, 1, paramData.length, 2).setValues(paramData);

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

  SpreadsheetApp.getActive().toast("Feuilles créées et mises en page !", "Setup", 3);
}


// =========================
// LIRE PARAMÈTRES
// =========================
function getParams() {

  const sheet = SpreadsheetApp.getActive().getSheetByName("Paramètres");
  const data = sheet.getRange(2, 2, 4, 1).getValues();

  return {
    num_vehicles: Number(data[0][0]) || 2,
    max_per_vehicle: Number(data[1][0]) || 35,
    start_id: data[2][0] ? String(data[2][0]) : "",
    end_id: data[3][0] ? String(data[3][0]) : ""
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
// APPEL API
// =========================
function callAPI(points, params) {

  const url = "https://tournees-api.onrender.com/optimize";

  const payload = {
    points: points,
    num_vehicles: params.num_vehicles,
    max_per_vehicle: params.max_per_vehicle,
    start_id: params.start_id,
    end_id: params.end_id
  };

  // Réveil du serveur (Render free tier s'endort)
  try {
    UrlFetchApp.fetch("https://tournees-api.onrender.com/", { muteHttpExceptions: true });
  } catch (e) {}
  Utilities.sleep(2000);

  const response = UrlFetchApp.fetch(url, {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  });

  const text = response.getContentText();
  const code = response.getResponseCode();

  if (code !== 200 || text.startsWith("<!")) {
    throw new Error("L'API n'est pas prête (code " + code + "). Réessayez dans 1 minute.");
  }

  return JSON.parse(text);
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

  var modeText = result.vroom_used ? "Vroom (affectation + séquencement)" : "K-Means (affectation) + Vroom (séquencement)";
  if (!result.vroom_used && result.vroom_error) {
    modeText += " | fallback: " + result.vroom_error;
  }
  sheet.getRange(infoRow + 1, 1).setValue("Mode");
  sheet.getRange(infoRow + 1, 2).setValue(modeText);
  sheet.getRange(infoRow + 1, 1).setFontWeight("bold");

  var km1 = result.tournee_1_km || 0;
  var km2 = result.tournee_2_km || 0;
  sheet.getRange(infoRow + 2, 1).setValue("Distance Tournée 1");
  sheet.getRange(infoRow + 2, 2).setValue(km1 + " km");
  sheet.getRange(infoRow + 2, 1).setFontWeight("bold");
  sheet.getRange(infoRow + 3, 1).setValue("Distance Tournée 2");
  sheet.getRange(infoRow + 3, 2).setValue(km2 + " km");
  sheet.getRange(infoRow + 3, 1).setFontWeight("bold");
  sheet.getRange(infoRow + 4, 1).setValue("Distance totale");
  sheet.getRange(infoRow + 4, 2).setValue(Math.round((km1 + km2) * 100) / 100 + " km");
  sheet.getRange(infoRow + 4, 1).setFontWeight("bold");
  sheet.getRange(infoRow + 4, 2).setFontWeight("bold");

  SpreadsheetApp.getActive().toast("Optimisation terminée !", "Résultat", 3);
}


// =========================
// LANCER L'OPTIMISATION
// =========================
function runOptimisation() {

  const params = getParams();
  const points = getPoints();

  if (points.length === 0) {
    SpreadsheetApp.getActive().toast("Aucun point sélectionné !", "Erreur", 3);
    return;
  }

  SpreadsheetApp.getActive().toast("Optimisation en cours... (" + points.length + " points)", "Info", 10);

  const result = callAPI(points, params);

  if (result.error) {
    SpreadsheetApp.getActive().toast("Erreur API : " + result.error, "Erreur", 5);
    return;
  }

  writeResult(result, params, points);

  var vroomInfo = result.vroom_used ? "Vroom direct" : "K-Means + Vroom (fallback)";
  SpreadsheetApp.getActive().toast("Terminé ! Mode : " + vroomInfo, "Succès", 10);
}


// =========================
// MENU PERSONNALISÉ
// =========================
function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu("Tournées")
    .addItem("Optimisation", "runOptimisation")
    .addSeparator()
    .addItem("Effacer tournées", "clearResults")
    .addItem("Réinitialiser la sélection", "resetSelection")
    .addToUi();
}


// =========================
// EFFACER TOURNÉES
// =========================
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