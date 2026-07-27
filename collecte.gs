function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Collecte')
    .addItem('Initialiser le classeur', 'initialiserClasseur')
    .addItem('Calculer la tournée', 'calculerTournee')
    .addItem('Réinitialiser la sélection', 'reinitialiserSelection')
    .addItem('Effacer la tournée', 'effacerTournee')
    .addSeparator()
    .addItem('Configurer la clé ORS', 'configurerCleORS')
    .addToUi();
}

function initialiserClasseur() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  initialiserClasseur_(spreadsheet);
  SpreadsheetApp.getUi().alert('Classeur initialisé et formaté.');
}

function configurerCleORS() {
  var ui = SpreadsheetApp.getUi();
  var response = ui.prompt('Clé API OpenRouteService', 'Collez votre clé API ORS Heigit.', ui.ButtonSet.OK_CANCEL);

  if (response.getSelectedButton() !== ui.Button.OK) {
    return;
  }

  var apiKey = String(response.getResponseText() || '').trim();
  if (!apiKey) {
    ui.alert('Aucune clé API n\'a été saisie.');
    return;
  }

  PropertiesService.getScriptProperties().setProperty('ORS_API_KEY', apiKey);
  ui.alert('Clé API ORS enregistrée avec succès.');
}

function calculerTournee() {
  var contexte = preparerCalculTournees_();
  var spreadsheet = contexte.spreadsheet;
  var pointsSelectionnes = contexte.pointsSelectionnes;
  var parametres = contexte.parametres;
  var profil = contexte.profil;
  var apiKey = contexte.apiKey;
  var tournees = null;

  tournees = planifierTournees_(pointsSelectionnes, parametres, profil, apiKey, {
    sheetNames: ['Tournee_J1', 'Tournee_J2']
  });

  effacerFeuillesTournees_(spreadsheet, ['Tournee_J1', 'Tournee_J2']);
  ecrireTourneesPlanifiees_(spreadsheet, tournees, null, ['Tournee_J1', 'Tournee_J2']);

  SpreadsheetApp.getUi().alert(
    tournees.length === 1
      ? 'Tournée calculée pour ' + pointsSelectionnes.length + ' horodateur(s) dans la feuille Tournee_J1.'
      : 'Deux tournées calculées pour ' + pointsSelectionnes.length + ' horodateur(s) dans les feuilles Tournee_J1 et Tournee_J2.'
  );
}

function preparerCalculTournees_() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var horodateursSheet = null;
  var parametresSheet = null;
  var apiKey = null;
  var pointsSelectionnes = null;
  var parametres = null;

  initialiserClasseur_(spreadsheet);
  horodateursSheet = spreadsheet.getSheetByName('Horodateurs');
  parametresSheet = spreadsheet.getSheetByName('Parametres');
  apiKey = PropertiesService.getScriptProperties().getProperty('ORS_API_KEY');

  if (!horodateursSheet) {
    throw new Error('La feuille "Horodateurs" est introuvable.');
  }

  if (!parametresSheet) {
    throw new Error('La feuille "Parametres" est introuvable.');
  }

  if (!apiKey) {
    throw new Error('La clé API ORS est absente. Utilisez le menu Collecte > Configurer la clé ORS.');
  }

  pointsSelectionnes = lireHorodateursSelectionnes_(horodateursSheet);
  if (pointsSelectionnes.length === 0) {
    throw new Error('Aucun horodateur sélectionné. Cochez au moins une ligne dans la colonne Selection.');
  }

  if (pointsSelectionnes.length > 60) {
    throw new Error('Le script gère au maximum 60 horodateurs sélectionnés à répartir sur 2 tournées.');
  }

  parametres = lireParametres_(parametresSheet);

  return {
    spreadsheet: spreadsheet,
    apiKey: apiKey,
    pointsSelectionnes: pointsSelectionnes,
    parametres: parametres,
    profil: parametres.profil || 'driving-car'
  };
}

function effacerTournee() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  effacerFeuillesTournees_(spreadsheet, ['Tournee_J1', 'Tournee_J2']);
  supprimerFeuillesSiExistantes_(spreadsheet, [
    'Tournee',
    'Comparatif_V2_J1',
    'Comparatif_V2_J2',
    'Comparatif_V3_J1',
    'Comparatif_V3_J2'
  ]);
}

function reinitialiserSelection() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = spreadsheet.getSheetByName('Horodateurs');
  var data = null;
  var headers = null;
  var rowCount = 0;
  var values = [];
  var i = 0;

  if (!sheet) {
    throw new Error('La feuille "Horodateurs" est introuvable.');
  }

  data = sheet.getDataRange().getValues();
  if (data.length < 2) {
    return;
  }

  headers = getHeaderMap_(data[0]);
  if (headers.selection === undefined) {
    throw new Error('La colonne "Selection" est introuvable dans la feuille "Horodateurs".');
  }

  rowCount = data.length - 1;
  for (i = 0; i < rowCount; i += 1) {
    values.push([false]);
  }

  sheet.getRange(2, headers.selection + 1, rowCount, 1).setValues(values);
}

function lireHorodateursSelectionnes_(sheet) {
  var data = sheet.getDataRange().getValues();
  if (data.length < 2) {
    return [];
  }

  var headers = getHeaderMap_(data[0]);
  var requiredHeaders = ['id', 'adresse', 'latitude', 'longitude', 'selection'];

  requiredHeaders.forEach(function(header) {
    if (headers[header] === undefined) {
      throw new Error('La colonne "' + toDisplayHeader_(header) + '" est introuvable dans la feuille "Horodateurs".');
    }
  });

  var points = [];

  for (var rowIndex = 1; rowIndex < data.length; rowIndex += 1) {
    var row = data[rowIndex];
    var selected = toBoolean_(row[headers.selection]);

    if (!selected) {
      continue;
    }

    var id = String(row[headers.id] || '').trim();
    var adresse = String(row[headers.adresse] || '').trim();
    var latitude = toNumber_(row[headers.latitude], 'Latitude', rowIndex + 1);
    var longitude = toNumber_(row[headers.longitude], 'Longitude', rowIndex + 1);

    if (!id) {
      throw new Error('ID manquant sur la ligne ' + (rowIndex + 1) + '.');
    }

    points.push({
      rowIndex: rowIndex + 1,
      id: id,
      adresse: adresse,
      latitude: latitude,
      longitude: longitude
    });
  }

  return points;
}

function lireParametres_(sheet) {
  var data = sheet.getDataRange().getValues();
  if (data.length < 2) {
    throw new Error('La feuille "Parametres" doit contenir un en-tête et une ligne de valeurs.');
  }

  var headers = getHeaderMap_(data[0]);
  var row = data[1];
  var requiredHeaders = ['depart_latitude', 'depart_longitude', 'arrivee_latitude', 'arrivee_longitude'];

  requiredHeaders.forEach(function(header) {
    if (headers[header] === undefined) {
      throw new Error('La colonne "' + toDisplayHeader_(header) + '" est introuvable dans la feuille "Parametres".');
    }
  });

  var profil = headers.profil_ors !== undefined ? String(row[headers.profil_ors] || '').trim() : 'driving-car';

  return {
    profil: profil || 'driving-car',
    depart: {
      id: headers.depart_id !== undefined ? String(row[headers.depart_id] || '').trim() || 'DEPART' : 'DEPART',
      adresse: headers.depart_adresse !== undefined ? String(row[headers.depart_adresse] || '').trim() : '',
      latitude: toNumber_(row[headers.depart_latitude], 'Depart_Latitude', 2),
      longitude: toNumber_(row[headers.depart_longitude], 'Depart_Longitude', 2)
    },
    arrivee: {
      id: headers.arrivee_id !== undefined ? String(row[headers.arrivee_id] || '').trim() || 'ARRIVEE' : 'ARRIVEE',
      adresse: headers.arrivee_adresse !== undefined ? String(row[headers.arrivee_adresse] || '').trim() : '',
      latitude: toNumber_(row[headers.arrivee_latitude], 'Arrivee_Latitude', 2),
      longitude: toNumber_(row[headers.arrivee_longitude], 'Arrivee_Longitude', 2)
    }
  };
}

function recupererMatriceORS_(points, profil, apiKey) {
  var url = 'https://api.openrouteservice.org/v2/matrix/' + encodeURIComponent(profil);
  var payload = {
    locations: points.map(function(point) {
      return [point.longitude, point.latitude];
    }),
    metrics: ['distance', 'duration'],
    units: 'm'
  };

  var response = UrlFetchApp.fetch(url, {
    method: 'post',
    contentType: 'application/json',
    headers: {
      Authorization: apiKey
    },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  });

  var status = response.getResponseCode();
  var text = response.getContentText();

  if (status < 200 || status >= 300) {
    throw new Error('Erreur ORS Matrix (' + status + ') : ' + text);
  }

  var json = JSON.parse(text);

  if (!json.durations || !json.distances) {
    throw new Error('Réponse ORS invalide : durées ou distances absentes.');
  }

  return {
    durations: json.durations,
    distances: json.distances
  };
}

function planifierTournees_(pointsSelectionnes, parametres, profil, apiKey, options) {
  var configuration = options || {};
  var groupes = configuration.groupes ? clonerGroupesPoints_(configuration.groupes) : repartirSelectionEnGroupes_(pointsSelectionnes, parametres);
  var sheetNames = configuration.sheetNames || ['Tournee_J1', 'Tournee_J2'];
  var tournees = [];
  var affinerRepartition = configuration.enableDistributionRefinement !== false;
  var i = 0;

  for (i = 0; i < groupes.length; i += 1) {
    if (groupes[i].length === 0) {
      continue;
    }

    tournees.push(
      calculerTourneePourGroupe_(
        groupes[i],
        parametres,
        profil,
        apiKey,
        sheetNames[i] || ('Tournee_J' + String(i + 1)),
        configuration
      )
    );
  }

  if (tournees.length === 2 && affinerRepartition) {
    tournees = raffinerDistributionEntreTournees_(tournees, parametres, profil, apiKey, configuration);
  }

  return tournees;
}

function repartirSelectionEnGroupes_(pointsSelectionnes, parametres) {
  var tailles = determinerTaillesTournees_(pointsSelectionnes.length);

  if (pointsSelectionnes.length <= 30) {
    return [pointsSelectionnes.slice()];
  }

  return repartirPointsEnDeuxGroupes_(pointsSelectionnes, parametres.depart, tailles);
}

function clonerGroupesPoints_(groupes) {
  return (groupes || []).map(function(groupe) {
    return groupe.slice();
  });
}

function determinerTaillesTournees_(totalPoints) {
  var tailleJ1 = 0;
  var tailleJ2 = 0;

  if (totalPoints <= 30) {
    return [totalPoints, 0];
  }

  tailleJ1 = Math.ceil(totalPoints / 2);
  tailleJ2 = totalPoints - tailleJ1;

  if (tailleJ1 > 30 || tailleJ2 > 30) {
    throw new Error('Impossible de répartir ' + totalPoints + ' horodateurs sur 2 tournées de 30 points maximum.');
  }

  return [tailleJ1, tailleJ2];
}

function calculerTourneePourGroupe_(groupePoints, parametres, profil, apiKey, sheetName, options) {
  var points = [parametres.depart].concat(groupePoints).concat([parametres.arrivee]);
  var matrix = recupererMatriceORS_(points, profil, apiKey);
  var ordreInterne = calculerOrdreOptimise_(points, matrix.durations, matrix.distances);
  var pairesLocales = construirePairesLocales_(points, matrix.durations, matrix.distances, 0, points.length - 1);
  var opportunitesInsertion = construireOpportunitesInsertion_(points, matrix.durations, 0, points.length - 1);
  var contexteUrbain = construireContexteUrbainTournee_(points, 0, points.length - 1, matrix.durations, matrix.distances);

  ordreInterne = reordonnerMicroSecteursContigus_(ordreInterne, matrix.durations, matrix.distances, 0, points.length - 1, pairesLocales, opportunitesInsertion, points, contexteUrbain);
  ordreInterne = ajusterOrdreParTrajectoireORS_(points, ordreInterne, matrix, profil, apiKey);

  return {
    sheetName: sheetName,
    points: points,
    ordreInterne: ordreInterne,
    matrix: matrix
  };
}

function raffinerDistributionEntreTournees_(tournees, parametres, profil, apiKey, options) {
  var meilleuresTournees = [tournees[0], tournees[1]];
  var meilleurScore = evaluerScoreTourneesCombinees_(meilleuresTournees);
  var candidatsJ1 = null;
  var candidatsJ2 = null;
  var pairesCandidates = null;
  var pointsJ1 = null;
  var pointsJ2 = null;
  var pointJ1 = null;
  var pointJ2 = null;
  var permutationJ1 = null;
  var permutationJ2 = null;
  var candidateTournees = null;
  var candidateScore = null;
  var meilleureCandidate = null;
  var paireCandidate = null;
  var iteration = 0;
  var deadline = Date.now() + 15000;
  var maxMovesParIteration = 2;
  var i = 0;

  while (iteration < 2 && Date.now() < deadline) {
    candidatsJ1 = extraireCandidatsFrontiereTournee_(meilleuresTournees[0], meilleuresTournees[1], 4);
    candidatsJ2 = extraireCandidatsFrontiereTournee_(meilleuresTournees[1], meilleuresTournees[0], 4);
    pairesCandidates = construirePairesPermutationCandidates_(candidatsJ1, candidatsJ2, 6);
    pointsJ1 = extrairePointsInterieursTournee_(meilleuresTournees[0]);
    pointsJ2 = extrairePointsInterieursTournee_(meilleuresTournees[1]);
    meilleureCandidate = null;

    for (i = 0; i < pairesCandidates.length && Date.now() < deadline; i += 1) {
      paireCandidate = pairesCandidates[i];
      pointJ1 = paireCandidate.pointJ1;
      pointJ2 = paireCandidate.pointJ2;
      permutationJ1 = remplacerPointDansGroupe_(pointsJ1, pointJ1, pointJ2);
      permutationJ2 = remplacerPointDansGroupe_(pointsJ2, pointJ2, pointJ1);

      candidateTournees = calculerCandidatDistributionEntreTournees_(
        permutationJ1,
        permutationJ2,
        meilleuresTournees,
        parametres,
        profil,
        apiKey,
        options
      );

      if (!candidateTournees) {
        continue;
      }

      candidateScore = evaluerScoreTourneesCombinees_(candidateTournees);

      if (estMeilleurScoreTourneesCombinees_(candidateScore, meilleurScore) && (!meilleureCandidate || estMeilleurScoreTourneesCombinees_(candidateScore, meilleureCandidate.score))) {
        meilleureCandidate = {
          tournees: candidateTournees,
          score: candidateScore
        }
      }
    }

    if (meilleurScore.streetSplitPenalty > 0 && pointsJ2.length < 30 && Date.now() < deadline) {
      for (i = 0; i < Math.min(candidatsJ1.length, maxMovesParIteration) && Date.now() < deadline; i += 1) {
        pointJ1 = candidatsJ1[i].point;
        permutationJ1 = enleverPointDuGroupe_(pointsJ1, pointJ1);
        permutationJ2 = pointsJ2.concat([pointJ1]);
        candidateTournees = calculerCandidatDistributionEntreTournees_(
          permutationJ1,
          permutationJ2,
          meilleuresTournees,
          parametres,
          profil,
          apiKey,
          options
        );

        if (!candidateTournees) {
          continue;
        }

        candidateScore = evaluerScoreTourneesCombinees_(candidateTournees);

        if (estMeilleurScoreTourneesCombinees_(candidateScore, meilleurScore) && (!meilleureCandidate || estMeilleurScoreTourneesCombinees_(candidateScore, meilleureCandidate.score))) {
          meilleureCandidate = {
            tournees: candidateTournees,
            score: candidateScore
          };
        }
      }
    }

    if (meilleurScore.streetSplitPenalty > 0 && pointsJ1.length < 30 && Date.now() < deadline) {
      for (i = 0; i < Math.min(candidatsJ2.length, maxMovesParIteration) && Date.now() < deadline; i += 1) {
        pointJ2 = candidatsJ2[i].point;
        permutationJ1 = pointsJ1.concat([pointJ2]);
        permutationJ2 = enleverPointDuGroupe_(pointsJ2, pointJ2);
        candidateTournees = calculerCandidatDistributionEntreTournees_(
          permutationJ1,
          permutationJ2,
          meilleuresTournees,
          parametres,
          profil,
          apiKey,
          options
        );

        if (!candidateTournees) {
          continue;
        }

        candidateScore = evaluerScoreTourneesCombinees_(candidateTournees);

        if (estMeilleurScoreTourneesCombinees_(candidateScore, meilleurScore) && (!meilleureCandidate || estMeilleurScoreTourneesCombinees_(candidateScore, meilleureCandidate.score))) {
          meilleureCandidate = {
            tournees: candidateTournees,
            score: candidateScore
          };
        }
      }
    }

    if (!meilleureCandidate) {
      break;
    }

    meilleuresTournees = meilleureCandidate.tournees;
    meilleurScore = meilleureCandidate.score;
    iteration += 1;
  }

  meilleuresTournees = finaliserDistributionEntreTournees_(meilleuresTournees, meilleurScore, parametres, profil, apiKey, options, deadline);
  return meilleuresTournees;
}

function extrairePointsInterieursTournee_(tournee) {
  return tournee.points.slice(1, tournee.points.length - 1);
}

function construirePairesPermutationCandidates_(candidatsJ1, candidatsJ2, maxCount) {
  var paires = [];
  var i = 0;
  var j = 0;

  for (i = 0; i < candidatsJ1.length; i += 1) {
    for (j = 0; j < candidatsJ2.length; j += 1) {
      paires.push({
        pointJ1: candidatsJ1[i].point,
        pointJ2: candidatsJ2[j].point,
        score: candidatsJ1[i].score + candidatsJ2[j].score
      });
    }
  }

  return paires.sort(function(a, b) {
    return b.score - a.score;
  }).slice(0, maxCount || 6);
}

function remplacerPointDansGroupe_(points, pointSortant, pointEntrant) {
  return points.map(function(point) {
    return point === pointSortant ? pointEntrant : point;
  });
}

function enleverPointDuGroupe_(points, pointASupprimer) {
  return points.filter(function(point) {
    return point !== pointASupprimer;
  });
}

function calculerCandidatDistributionEntreTournees_(pointsJ1, pointsJ2, tourneesReference, parametres, profil, apiKey, options) {
  try {
    return [
      calculerTourneePourGroupe_(pointsJ1, parametres, profil, apiKey, tourneesReference[0].sheetName, options),
      calculerTourneePourGroupe_(pointsJ2, parametres, profil, apiKey, tourneesReference[1].sheetName, options)
    ];
  } catch (error) {
    return null;
  }
}

function finaliserDistributionEntreTournees_(tournees, scoreReference, parametres, profil, apiKey, options, deadline) {
  var pointsJ1 = extrairePointsInterieursTournee_(tournees[0]);
  var pointsJ2 = extrairePointsInterieursTournee_(tournees[1]);
  var contexteBassins = construireContexteBassinsGroupes_(pointsJ1.concat(pointsJ2));
  var ruesReparties = null;
  var candidateTournees = null;
  var candidateScore = null;

  if (Date.now() >= deadline) {
    return tournees;
  }

  ruesReparties = listerRuesRepartiesEntreGroupes_(pointsJ1, pointsJ2);
  if (ruesReparties.length === 0) {
    return tournees;
  }

  consoliderRuesReparties_(pointsJ1, pointsJ2, 30, 30, parametres.depart, 2, 3, contexteBassins);

  if (Date.now() >= deadline) {
    return tournees;
  }

  candidateTournees = calculerCandidatDistributionEntreTournees_(pointsJ1, pointsJ2, tournees, parametres, profil, apiKey, options);
  if (!candidateTournees) {
    return tournees;
  }

  candidateScore = evaluerScoreTourneesCombinees_(candidateTournees);
  return estMeilleurScoreTourneesCombinees_(candidateScore, scoreReference) ? candidateTournees : tournees;
}

function evaluerTourneeCalculee_(tournee) {
  var startIndex = 0;
  var endIndex = tournee.points.length - 1;
  var pairesLocales = construirePairesLocales_(tournee.points, tournee.matrix.durations, tournee.matrix.distances, startIndex, endIndex);
  var opportunitesInsertion = construireOpportunitesInsertion_(tournee.points, tournee.matrix.durations, startIndex, endIndex);
  var contexteUrbain = construireContexteUrbainTournee_(tournee.points, startIndex, endIndex, tournee.matrix.durations, tournee.matrix.distances);

  return evaluerParcours_(tournee.ordreInterne, tournee.matrix.durations, tournee.matrix.distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, tournee.points, contexteUrbain);
}

function evaluerScoreTourneesCombinees_(tournees) {
  var evaluations = tournees.map(function(tournee) {
    return evaluerTourneeCalculee_(tournee);
  });
  var duration = 0;
  var distance = 0;
  var opportunityPenalty = 0;
  var localPenalty = 0;
  var balancePenalty = 0;
  var streetSplitPenalty = 0;
  var sectorSplitPenalty = 0;
  var sectorReentryPenalty = 0;
  var appendixMidRoutePenalty = 0;
  var proximityDetourPenalty = 0;
  var twoStepLookaheadPenalty = 0;
  var deferredPassagePenalty = 0;
  var groupes = [];
  var i = 0;

  for (i = 0; i < evaluations.length; i += 1) {
    duration += evaluations[i].duration;
    distance += evaluations[i].distance;
    opportunityPenalty += evaluations[i].opportunityPenalty;
    localPenalty += evaluations[i].localPenalty;
    sectorReentryPenalty += evaluations[i].sectorReentryPenalty || 0;
    appendixMidRoutePenalty += evaluations[i].appendixMidRoutePenalty || 0;
    proximityDetourPenalty += evaluations[i].proximityDetourPenalty || 0;
    twoStepLookaheadPenalty += evaluations[i].twoStepLookaheadPenalty || 0;
    deferredPassagePenalty += evaluations[i].deferredPassagePenalty || 0;
  }

  if (evaluations.length === 2) {
    balancePenalty = Math.abs(evaluations[0].duration - evaluations[1].duration);
  }

  streetSplitPenalty = calculerPenaliteSeparationRuesTournees_(tournees);
  groupes = (tournees || []).map(function(tournee) {
    return extrairePointsInterieursTournee_(tournee);
  });
  sectorSplitPenalty = calculerPenaliteSeparationBassinsGroupes_(groupes, construireContexteBassinsGroupes_([].concat.apply([], groupes)));

  return {
    duration: duration,
    distance: distance,
    balancePenalty: balancePenalty,
    streetSplitPenalty: streetSplitPenalty,
    sectorSplitPenalty: sectorSplitPenalty,
    opportunityPenalty: opportunityPenalty,
    localPenalty: localPenalty,
    sectorReentryPenalty: sectorReentryPenalty,
    appendixMidRoutePenalty: appendixMidRoutePenalty,
    proximityDetourPenalty: proximityDetourPenalty,
    twoStepLookaheadPenalty: twoStepLookaheadPenalty,
    deferredPassagePenalty: deferredPassagePenalty
  };
}

function calculerPenaliteSeparationRuesTournees_(tournees) {
  return calculerPenaliteSeparationRuesGroupes_((tournees || []).map(function(tournee) {
    return extrairePointsInterieursTournee_(tournee);
  }));
}

function estMeilleurScoreTourneesCombinees_(candidat, reference) {
  var toleranceDuree = 60;
  var toleranceDistance = 200;
  var toleranceBalance = 120;
  var toleranceSeparationRue = 0.5;
  var toleranceSeparationBassin = 0.5;
  var toleranceOpportunite = 25;
  var toleranceLocale = 0.25;
  var toleranceReentreeBassin = 0.35;
  var toleranceAppendice = 0.35;
  var tolerancePassageDiffere = 0.6;
  var poidsDetourProximite = 12;
  var poidsLookahead = 10;
  var poidsBacktracking = 8;
  var dureePondereeCandidat = 0;
  var dureePondereeReference = 0;

  if (!reference) {
    return true;
  }

  dureePondereeCandidat = candidat.duration + (candidat.proximityDetourPenalty || 0) * poidsDetourProximite + (candidat.twoStepLookaheadPenalty || 0) * poidsLookahead + (candidat.backtrackingPenalty || 0) * poidsBacktracking;
  dureePondereeReference = reference.duration + (reference.proximityDetourPenalty || 0) * poidsDetourProximite + (reference.twoStepLookaheadPenalty || 0) * poidsLookahead + (reference.backtrackingPenalty || 0) * poidsBacktracking;

  if (dureePondereeCandidat + toleranceDuree < dureePondereeReference) {
    return true;
  }

  if (dureePondereeReference + toleranceDuree < dureePondereeCandidat) {
    return false;
  }

  if (candidat.streetSplitPenalty + toleranceSeparationRue < reference.streetSplitPenalty) {
    return true;
  }

  if (reference.streetSplitPenalty + toleranceSeparationRue < candidat.streetSplitPenalty) {
    return false;
  }

  if (candidat.sectorSplitPenalty + toleranceSeparationBassin < reference.sectorSplitPenalty) {
    return true;
  }

  if (reference.sectorSplitPenalty + toleranceSeparationBassin < candidat.sectorSplitPenalty) {
    return false;
  }

  if (candidat.deferredPassagePenalty + tolerancePassageDiffere < reference.deferredPassagePenalty) {
    return true;
  }

  if (reference.deferredPassagePenalty + tolerancePassageDiffere < candidat.deferredPassagePenalty) {
    return false;
  }

  if (candidat.distance + toleranceDistance < reference.distance) {
    return true;
  }

  if (reference.distance + toleranceDistance < candidat.distance) {
    return false;
  }

  if (candidat.balancePenalty + toleranceBalance < reference.balancePenalty) {
    return true;
  }

  if (reference.balancePenalty + toleranceBalance < candidat.balancePenalty) {
    return false;
  }

  if (candidat.sectorReentryPenalty + toleranceReentreeBassin < reference.sectorReentryPenalty) {
    return true;
  }

  if (reference.sectorReentryPenalty + toleranceReentreeBassin < candidat.sectorReentryPenalty) {
    return false;
  }

  if (candidat.appendixMidRoutePenalty + toleranceAppendice < reference.appendixMidRoutePenalty) {
    return true;
  }

  if (reference.appendixMidRoutePenalty + toleranceAppendice < candidat.appendixMidRoutePenalty) {
    return false;
  }

  if (candidat.opportunityPenalty + toleranceOpportunite < reference.opportunityPenalty) {
    return true;
  }

  if (reference.opportunityPenalty + toleranceOpportunite < candidat.opportunityPenalty) {
    return false;
  }

  if (candidat.localPenalty + toleranceLocale < reference.localPenalty) {
    return true;
  }

  if (reference.localPenalty + toleranceLocale < candidat.localPenalty) {
    return false;
  }

  if (dureePondereeCandidat !== dureePondereeReference) {
    return dureePondereeCandidat < dureePondereeReference;
  }

  if (candidat.streetSplitPenalty !== reference.streetSplitPenalty) {
    return candidat.streetSplitPenalty < reference.streetSplitPenalty;
  }

  if (candidat.sectorSplitPenalty !== reference.sectorSplitPenalty) {
    return candidat.sectorSplitPenalty < reference.sectorSplitPenalty;
  }

  if (candidat.deferredPassagePenalty !== reference.deferredPassagePenalty) {
    return candidat.deferredPassagePenalty < reference.deferredPassagePenalty;
  }

  if (candidat.distance !== reference.distance) {
    return candidat.distance < reference.distance;
  }

  if (candidat.balancePenalty !== reference.balancePenalty) {
    return candidat.balancePenalty < reference.balancePenalty;
  }

  if (candidat.sectorReentryPenalty !== reference.sectorReentryPenalty) {
    return candidat.sectorReentryPenalty < reference.sectorReentryPenalty;
  }

  if (candidat.appendixMidRoutePenalty !== reference.appendixMidRoutePenalty) {
    return candidat.appendixMidRoutePenalty < reference.appendixMidRoutePenalty;
  }

  if (candidat.opportunityPenalty !== reference.opportunityPenalty) {
    return candidat.opportunityPenalty < reference.opportunityPenalty;
  }

  return candidat.localPenalty < reference.localPenalty;
}

function extraireCandidatsFrontiereTournee_(tournee, autreTournee, maxCount) {
  var pointsTournee = extrairePointsInterieursTournee_(tournee);
  var pointsAutreTournee = extrairePointsInterieursTournee_(autreTournee);
  var orderedPointIndexes = [0].concat(tournee.ordreInterne).concat([tournee.points.length - 1]);
  var centreTournee = calculerCentreGroupe_(pointsTournee);
  var centreAutreTournee = calculerCentreGroupe_(pointsAutreTournee);
  var candidats = [];
  var point = null;
  var nodeIndex = 0;
  var previousIndex = 0;
  var nextIndex = 0;
  var removalDelta = 0;
  var i = 0;

  for (i = 1; i < orderedPointIndexes.length - 1; i += 1) {
    nodeIndex = orderedPointIndexes[i];
    point = tournee.points[nodeIndex];
    previousIndex = orderedPointIndexes[i - 1];
    nextIndex = orderedPointIndexes[i + 1];
    removalDelta = safeCost_(tournee.matrix.durations[previousIndex][nodeIndex])
      + safeCost_(tournee.matrix.durations[nodeIndex][nextIndex])
      - safeCost_(tournee.matrix.durations[previousIndex][nextIndex]);

    candidats.push({
      point: point,
      score: evaluerPointFrontiere_(point, pointsTournee, pointsAutreTournee, centreTournee, centreAutreTournee, removalDelta)
    });
  }

  return candidats.sort(function(a, b) {
    return b.score - a.score;
  }).slice(0, maxCount || 4);
}

function evaluerPointFrontiere_(point, pointsTournee, pointsAutreTournee, centreTournee, centreAutreTournee, removalDelta) {
  var distanceCentreTournee = calculerDistanceMetres_(point, centreTournee);
  var distanceCentreAutreTournee = calculerDistanceMetres_(point, centreAutreTournee);
  var distanceProcheTournee = distanceMinimaleAuGroupe_(point, pointsTournee, point);
  var distanceProcheAutreTournee = distanceMinimaleAuGroupe_(point, pointsAutreTournee, null);
  var pointsMemeRueTournee = compterPointsMemeRueDansGroupe_(point, pointsTournee, point);
  var pointsMemeRueAutreTournee = compterPointsMemeRueDansGroupe_(point, pointsAutreTournee, null);
  var scoreFrontiere = removalDelta;

  if (isFinite(distanceProcheAutreTournee)) {
    scoreFrontiere += Math.max(0, 450 - distanceProcheAutreTournee) * 0.6;
  }

  if (isFinite(distanceProcheTournee)) {
    scoreFrontiere += distanceProcheTournee * 0.2;
  }

  scoreFrontiere += Math.max(0, pointsMemeRueAutreTournee - pointsMemeRueTournee) * 180;
  scoreFrontiere += pointsMemeRueAutreTournee * 45;
  scoreFrontiere -= pointsMemeRueTournee * 35;
  scoreFrontiere += Math.max(0, distanceCentreTournee - distanceCentreAutreTournee) * 0.25;
  return scoreFrontiere;
}

function distanceMinimaleAuGroupe_(point, groupe, pointExclu) {
  var meilleureDistance = Number.POSITIVE_INFINITY;
  var distance = 0;
  var i = 0;

  if (!groupe || groupe.length === 0) {
    return Number.POSITIVE_INFINITY;
  }

  for (i = 0; i < groupe.length; i += 1) {
    if (pointExclu && groupe[i] === pointExclu) {
      continue;
    }

    distance = calculerDistanceMetres_(point, groupe[i]);
    if (distance < meilleureDistance) {
      meilleureDistance = distance;
    }
  }

  return meilleureDistance;
}

function compterPointsMemeRueDansGroupe_(point, groupe, pointExclu) {
  var rue = canoniserNomRue_(point && point.adresse);
  var total = 0;
  var i = 0;

  if (!rue || !groupe || groupe.length === 0) {
    return 0;
  }

  for (i = 0; i < groupe.length; i += 1) {
    if (pointExclu && groupe[i] === pointExclu) {
      continue;
    }

    if (canoniserNomRue_(groupe[i] && groupe[i].adresse) === rue) {
      total += 1;
    }
  }

  return total;
}

function calculerPenaliteSeparationRuesGroupes_(groupes) {
  var repartition = {};
  var penalite = 0;
  var groupe = null;
  var point = null;
  var rue = '';
  var compteurs = null;
  var i = 0;
  var j = 0;

  for (i = 0; i < (groupes || []).length; i += 1) {
    groupe = groupes[i] || [];

    for (j = 0; j < groupe.length; j += 1) {
      point = groupe[j];
      rue = canoniserNomRue_(point && point.adresse);

      if (!rue) {
        continue;
      }

      if (!repartition[rue]) {
        repartition[rue] = [];
      }

      repartition[rue][i] = (repartition[rue][i] || 0) + 1;
    }
  }

  Object.keys(repartition).forEach(function(cleRue) {
    compteurs = repartition[cleRue].filter(function(count) {
      return count > 0;
    });

    if (compteurs.length > 1) {
      penalite += Math.min.apply(null, compteurs);
    }
  });

  return penalite;
}

function calculerPenaliteDetourProximite_(route, contexteUrbain) {
  var penalite = 0;
  var arcs = contexteUrbain ? contexteUrbain.proximityDetourArcs : null;
  var i = 0;
  var cleArc = '';

  if (!arcs) {
    return 0;
  }

  for (i = 0; i < route.length - 1; i += 1) {
    cleArc = construireCleArc_(route[i], route[i + 1]);
    if (arcs[cleArc]) {
      penalite += arcs[cleArc];
    }
  }

  return penalite;
}

function calculerPenaliteLookaheadDeuxPoints_(route, durations, distances, startIndex, endIndex, points) {
  var parcours = [startIndex].concat(route).concat([endIndex]);
  var penalite = 0;
  var indexA = 0;
  var indexB = 0;
  var indexC = 0;
  var indexD = 0;
  var distanceGeo = 0;
  var rueB = '';
  var rueC = '';
  var memeRue = false;
  var coutActuel = 0;
  var coutAlternatif = 0;
  var distanceActuelle = 0;
  var diff = 0;
  var poids = 0;
  var i = 0;

  if (!route || route.length < 2 || !points) {
    return 0;
  }

  for (i = 1; i < parcours.length - 2; i += 1) {
    indexA = parcours[i - 1];
    indexB = parcours[i];
    indexC = parcours[i + 1];
    indexD = parcours[i + 2];
    distanceGeo = calculerDistanceMetres_(points[indexB], points[indexC]);
    rueB = canoniserNomRue_(points[indexB] && points[indexB].adresse);
    rueC = canoniserNomRue_(points[indexC] && points[indexC].adresse);
    memeRue = !!rueB && rueB === rueC;

    if (distanceGeo > (memeRue ? 380 : 250)) {
      continue;
    }

    coutActuel = safeCost_(durations[indexA][indexB]) + safeCost_(durations[indexB][indexC]) + safeCost_(durations[indexC][indexD]);
    coutAlternatif = safeCost_(durations[indexA][indexC]) + safeCost_(durations[indexC][indexB]) + safeCost_(durations[indexB][indexD]);
    diff = coutActuel - coutAlternatif;

    if (diff <= (memeRue ? 25 : 20)) {
      continue;
    }

    poids = Math.min(8, diff / (memeRue ? 30 : 40));
    distanceActuelle = safeCost_(distances[indexB][indexC]);

    if (distanceActuelle > (memeRue ? 700 : 420)) {
      poids += Math.min(3, (distanceActuelle - (memeRue ? 700 : 420)) / 170);
    }

    if (distanceGeo <= 75) {
      poids += 0.8;
    }

    if (memeRue) {
      poids *= 1.25;
    }

    penalite += poids;
  }

  return penalite;
}

function listerRuesRepartiesEntreGroupes_(groupeJ1, groupeJ2) {
  var repartition = {};
  var point = null;
  var rue = '';
  var i = 0;
  var resultats = [];

  for (i = 0; i < (groupeJ1 || []).length; i += 1) {
    point = groupeJ1[i];
    rue = canoniserNomRue_(point && point.adresse);

    if (!rue) {
      continue;
    }

    if (!repartition[rue]) {
      repartition[rue] = {
        rue: rue,
        pointsJ1: [],
        pointsJ2: []
      };
    }

    repartition[rue].pointsJ1.push(point);
  }

  for (i = 0; i < (groupeJ2 || []).length; i += 1) {
    point = groupeJ2[i];
    rue = canoniserNomRue_(point && point.adresse);

    if (!rue) {
      continue;
    }

    if (!repartition[rue]) {
      repartition[rue] = {
        rue: rue,
        pointsJ1: [],
        pointsJ2: []
      };
    }

    repartition[rue].pointsJ2.push(point);
  }

  Object.keys(repartition).forEach(function(cleRue) {
    if (repartition[cleRue].pointsJ1.length > 0 && repartition[cleRue].pointsJ2.length > 0) {
      resultats.push(repartition[cleRue]);
    }
  });

  return resultats.sort(function(a, b) {
    return Math.min(b.pointsJ1.length, b.pointsJ2.length) - Math.min(a.pointsJ1.length, a.pointsJ2.length);
  });
}

function enleverPointsDuGroupe_(points, pointsASupprimer) {
  return points.filter(function(point) {
    return pointsASupprimer.indexOf(point) === -1;
  });
}

function consoliderRuesReparties_(groupeJ1, groupeJ2, tailleMaxJ1, tailleMaxJ2, depot, maxIterations, maxRuesTeste, contexteBassins) {
  var amelioration = true;
  var iteration = 0;
  var ruesReparties = null;
  var scoreReference = 0;
  var meilleureOption = null;
  var candidatJ1 = null;
  var candidatJ2 = null;
  var scoreCandidat = 0;
  var delta = 0;
  var i = 0;

  maxIterations = maxIterations || 3;
  maxRuesTeste = maxRuesTeste || 4;

  while (amelioration && iteration < maxIterations) {
    amelioration = false;
    meilleureOption = null;
    ruesReparties = listerRuesRepartiesEntreGroupes_(groupeJ1, groupeJ2).slice(0, maxRuesTeste);

    if (ruesReparties.length === 0) {
      break;
    }

    scoreReference = evaluerRepartitionGeographique_(groupeJ1, groupeJ2, depot, contexteBassins);

    for (i = 0; i < ruesReparties.length; i += 1) {
      if (groupeJ1.length + ruesReparties[i].pointsJ2.length <= tailleMaxJ1) {
        candidatJ1 = groupeJ1.concat(ruesReparties[i].pointsJ2);
        candidatJ2 = enleverPointsDuGroupe_(groupeJ2, ruesReparties[i].pointsJ2);
        scoreCandidat = evaluerRepartitionGeographique_(candidatJ1, candidatJ2, depot, contexteBassins);
        delta = scoreReference - scoreCandidat;

        if (delta > 0 && (!meilleureOption || delta > meilleureOption.delta)) {
          meilleureOption = {
            groupeJ1: candidatJ1,
            groupeJ2: candidatJ2,
            delta: delta
          };
        }
      }

      if (groupeJ2.length + ruesReparties[i].pointsJ1.length <= tailleMaxJ2) {
        candidatJ1 = enleverPointsDuGroupe_(groupeJ1, ruesReparties[i].pointsJ1);
        candidatJ2 = groupeJ2.concat(ruesReparties[i].pointsJ1);
        scoreCandidat = evaluerRepartitionGeographique_(candidatJ1, candidatJ2, depot, contexteBassins);
        delta = scoreReference - scoreCandidat;

        if (delta > 0 && (!meilleureOption || delta > meilleureOption.delta)) {
          meilleureOption = {
            groupeJ1: candidatJ1,
            groupeJ2: candidatJ2,
            delta: delta
          };
        }
      }
    }

    if (meilleureOption) {
      remplacerContenuGroupe_(groupeJ1, meilleureOption.groupeJ1);
      remplacerContenuGroupe_(groupeJ2, meilleureOption.groupeJ2);
      amelioration = true;
    }

    iteration += 1;
  }
}

function remplacerContenuGroupe_(groupeCible, nouveauContenu) {
  groupeCible.length = 0;
  Array.prototype.push.apply(groupeCible, nouveauContenu);
}

function repartirPointsEnDeuxGroupes_(points, depot, tailles) {
  var contexteBassins = construireContexteBassinsGroupes_(points);
  var repartitionBassins = essayerRepartirBassinsEnDeuxGroupes_(contexteBassins, depot, tailles);
  var graines = trouverGrainesRepartition_(points, depot);
  var groupeJ1 = repartitionBassins ? repartitionBassins[0] : [graines[0]];
  var groupeJ2 = repartitionBassins ? repartitionBassins[1] : [graines[1]];
  var restants = repartitionBassins ? [] : points.filter(function(point) {
    return point !== graines[0] && point !== graines[1];
  });
  var i = 0;

  restants.sort(function(pointA, pointB) {
    var diffA = Math.abs(
      calculerDistanceMetres_(pointA, graines[0]) - calculerDistanceMetres_(pointA, graines[1])
    );
    var diffB = Math.abs(
      calculerDistanceMetres_(pointB, graines[0]) - calculerDistanceMetres_(pointB, graines[1])
    );

    return diffB - diffA;
  });

  for (i = 0; i < restants.length; i += 1) {
    affecterPointAuMeilleurGroupe_(restants[i], groupeJ1, groupeJ2, depot, tailles[0], tailles[1], contexteBassins);
  }

  ameliorerRepartitionParEchanges_(groupeJ1, groupeJ2, depot, contexteBassins);
  consoliderRuesReparties_(groupeJ1, groupeJ2, tailles[0], tailles[1], depot, 3, 4, contexteBassins);

  return [groupeJ1, groupeJ2];
}

function trouverGrainesRepartition_(points, depot) {
  var meilleureDistance = -1;
  var meilleurePaire = null;
  var i = 0;
  var j = 0;
  var distance = 0;
  var pointLePlusEloigne = null;

  if (points.length === 1) {
    return [points[0], points[0]];
  }

  for (i = 0; i < points.length - 1; i += 1) {
    for (j = i + 1; j < points.length; j += 1) {
      distance = calculerDistanceMetres_(points[i], points[j]);

      if (distance > meilleureDistance) {
        meilleureDistance = distance;
        meilleurePaire = [points[i], points[j]];
      }
    }
  }

  if (meilleurePaire) {
    return meilleurePaire;
  }

  pointLePlusEloigne = points.slice().sort(function(pointA, pointB) {
    return calculerDistanceMetres_(depot, pointB) - calculerDistanceMetres_(depot, pointA);
  });

  return [pointLePlusEloigne[0], pointLePlusEloigne[1] || pointLePlusEloigne[0]];
}

function affecterPointAuMeilleurGroupe_(point, groupeJ1, groupeJ2, depot, tailleMaxJ1, tailleMaxJ2, contexteBassins) {
  var scoreJ1 = 0;
  var scoreJ2 = 0;

  if (groupeJ1.length >= tailleMaxJ1) {
    groupeJ2.push(point);
    return;
  }

  if (groupeJ2.length >= tailleMaxJ2) {
    groupeJ1.push(point);
    return;
  }

  scoreJ1 = evaluerInsertionGeographique_(point, groupeJ1, groupeJ2, depot, contexteBassins);
  scoreJ2 = evaluerInsertionGeographique_(point, groupeJ2, groupeJ1, depot, contexteBassins);

  if (scoreJ1 < scoreJ2) {
    groupeJ1.push(point);
    return;
  }

  if (scoreJ2 < scoreJ1) {
    groupeJ2.push(point);
    return;
  }

  if (groupeJ1.length <= groupeJ2.length) {
    groupeJ1.push(point);
  } else {
    groupeJ2.push(point);
  }
}

function evaluerInsertionGeographique_(point, groupe, autreGroupe, depot, contexteBassins) {
  var centre = null;
  var distanceCentre = 0;
  var distanceProche = 0;
  var pointsMemeRueGroupe = compterPointsMemeRueDansGroupe_(point, groupe, null);
  var pointsMemeRueAutreGroupe = compterPointsMemeRueDansGroupe_(point, autreGroupe, null);
  var pointsMemeBassinGroupe = compterPointsMemeBassinDansGroupe_(point, groupe, contexteBassins, null);
  var pointsMemeBassinAutreGroupe = compterPointsMemeBassinDansGroupe_(point, autreGroupe, contexteBassins, null);
  var score = 0;

  if (groupe.length === 0) {
    score = calculerDistanceMetres_(point, depot);
  } else {
    centre = calculerCentreGroupe_(groupe);
    distanceCentre = calculerDistanceMetres_(point, centre);
    distanceProche = distanceMinimaleAuGroupe_(point, groupe, null);

    score = distanceProche * 0.55 + distanceCentre * 0.30 + calculerDistanceMetres_(point, depot) * 0.15;
  }

  score -= pointsMemeRueGroupe * 120;
  score += Math.max(0, pointsMemeRueAutreGroupe - pointsMemeRueGroupe) * 90;
  score -= pointsMemeBassinGroupe * 160;
  score += Math.max(0, pointsMemeBassinAutreGroupe - pointsMemeBassinGroupe) * 130;

  return score;
}

function ameliorerRepartitionParEchanges_(groupeJ1, groupeJ2, depot, contexteBassins) {
  var amelioration = true;
  var iteration = 0;
  var meilleurDelta = 0;
  var meilleurI = -1;
  var meilleurJ = -1;
  var scoreReference = 0;
  var scoreCandidat = 0;
  var candidatJ1 = null;
  var candidatJ2 = null;
  var tampon = null;
  var i = 0;
  var j = 0;

  while (amelioration && iteration < 4) {
    amelioration = false;
    meilleurDelta = 0;
    meilleurI = -1;
    meilleurJ = -1;
    scoreReference = evaluerRepartitionGeographique_(groupeJ1, groupeJ2, depot, contexteBassins);

    for (i = 0; i < groupeJ1.length; i += 1) {
      for (j = 0; j < groupeJ2.length; j += 1) {
        candidatJ1 = groupeJ1.slice();
        candidatJ2 = groupeJ2.slice();
        tampon = candidatJ1[i];
        candidatJ1[i] = candidatJ2[j];
        candidatJ2[j] = tampon;
        scoreCandidat = evaluerRepartitionGeographique_(candidatJ1, candidatJ2, depot, contexteBassins);

        if (scoreReference - scoreCandidat > meilleurDelta) {
          meilleurDelta = scoreReference - scoreCandidat;
          meilleurI = i;
          meilleurJ = j;
        }
      }
    }

    if (meilleurI !== -1 && meilleurJ !== -1) {
      tampon = groupeJ1[meilleurI];
      groupeJ1[meilleurI] = groupeJ2[meilleurJ];
      groupeJ2[meilleurJ] = tampon;
      amelioration = true;
    }

    iteration += 1;
  }
}

function evaluerRepartitionGeographique_(groupeJ1, groupeJ2, depot, contexteBassins) {
  return evaluerGroupeGeographique_(groupeJ1, depot)
    + evaluerGroupeGeographique_(groupeJ2, depot)
    + calculerPenaliteSeparationRuesGroupes_([groupeJ1, groupeJ2]) * 180
    + calculerPenaliteSeparationBassinsGroupes_([groupeJ1, groupeJ2], contexteBassins) * 260;
}

function evaluerGroupeGeographique_(groupe, depot) {
  var centre = calculerCentreGroupe_(groupe);
  var score = 0;
  var distanceProche = 0;
  var i = 0;

  if (groupe.length === 0) {
    return 0;
  }

  for (i = 0; i < groupe.length; i += 1) {
    distanceProche = distanceMinimaleAuGroupe_(groupe[i], groupe, groupe[i]);
    score += calculerDistanceMetres_(groupe[i], centre);
    score += isFinite(distanceProche) ? distanceProche * 0.35 : 0;
    score += calculerDistanceMetres_(groupe[i], depot) * 0.10;
  }

  return score;
}

function calculerCentreGroupe_(groupe) {
  var latitude = 0;
  var longitude = 0;
  var i = 0;

  if (groupe.length === 0) {
    return {
      latitude: 0,
      longitude: 0
    };
  }

  for (i = 0; i < groupe.length; i += 1) {
    latitude += groupe[i].latitude;
    longitude += groupe[i].longitude;
  }

  return {
    latitude: latitude / groupe.length,
    longitude: longitude / groupe.length
  };
}

function construireClePoint_(point) {
  if (!point) {
    return '';
  }

  if (point.id !== undefined && point.id !== null && point.id !== '') {
    return String(point.id);
  }

  return roundTo_(point.latitude || 0, 6) + ',' + roundTo_(point.longitude || 0, 6);
}

function construireContexteBassinsGroupes_(points) {
  var bassins = construireBassinsUrbains_(points || []);
  var bassinParPoint = {};
  var pointsLength = (points || []).length;
  var i = 0;
  var clePoint = '';

  for (i = 0; i < bassins.length; i += 1) {
    for (var j = 0; j < bassins[i].points.length; j += 1) {
      clePoint = construireClePoint_(bassins[i].points[j]);
      if (clePoint) {
        bassinParPoint[clePoint] = i;
      }
    }
  }

  return {
    bassins: bassins,
    bassinParPoint: bassinParPoint,
    pointsLength: pointsLength
  };
}

function construireBassinsUrbains_(points) {
  var graphe = null;
  var composantes = null;
  var bassins = [];
  var i = 0;
  var pointsBassin = null;
  var centre = null;
  var appendices = null;

  if (!points || points.length === 0) {
    return [];
  }

  graphe = construireGrapheUrbain_(points);
  composantes = extraireComposantesUrbaines_(graphe);

  for (i = 0; i < composantes.length; i += 1) {
    pointsBassin = composantes[i].map(function(indexPoint) {
      return points[indexPoint];
    });
    centre = calculerCentreGroupe_(pointsBassin);
    appendices = determinerAppendicesBassinUrbain_(pointsBassin, centre);
    bassins.push({
      id: i,
      points: pointsBassin,
      centre: centre,
      appendices: appendices
    });
  }

  return bassins.sort(function(a, b) {
    return b.points.length - a.points.length;
  });
}

function construireGrapheUrbain_(points) {
  var graphe = [];
  var voisins = [];
  var i = 0;
  var j = 0;
  var distance = 0;
  var plusProches = [];

  for (i = 0; i < points.length; i += 1) {
    graphe[i] = [];
    plusProches = [];

    for (j = 0; j < points.length; j += 1) {
      if (i === j) {
        continue;
      }

      distance = calculerDistanceMetres_(points[i], points[j]);
      plusProches.push({
        index: j,
        distance: distance
      });

      if (distance <= 240) {
        graphe[i].push(j);
      }
    }

    plusProches.sort(function(a, b) {
      return a.distance - b.distance;
    });
    voisins[i] = plusProches.slice(0, 4).map(function(item) {
      return item.index;
    });
  }

  for (i = 0; i < points.length; i += 1) {
    for (j = i + 1; j < points.length; j += 1) {
      distance = calculerDistanceMetres_(points[i], points[j]);

      if (distance <= 340 && voisins[i].indexOf(j) !== -1 && voisins[j].indexOf(i) !== -1) {
        ajouterUnique_(graphe[i], j);
        ajouterUnique_(graphe[j], i);
      }
    }
  }

  return graphe;
}

function extraireComposantesUrbaines_(graphe) {
  var visites = {};
  var composantes = [];
  var pile = [];
  var composante = null;
  var indexCourant = 0;
  var voisin = 0;
  var i = 0;
  var j = 0;

  for (i = 0; i < graphe.length; i += 1) {
    if (visites[i]) {
      continue;
    }

    pile = [i];
    composante = [];
    visites[i] = true;

    while (pile.length > 0) {
      indexCourant = pile.pop();
      composante.push(indexCourant);

      for (j = 0; j < graphe[indexCourant].length; j += 1) {
        voisin = graphe[indexCourant][j];
        if (!visites[voisin]) {
          visites[voisin] = true;
          pile.push(voisin);
        }
      }
    }

    composantes.push(composante);
  }

  return composantes;
}

function determinerAppendicesBassinUrbain_(pointsBassin, centre) {
  var appendices = [];
  var voisins = 0;
  var distanceCentre = 0;
  var i = 0;
  var j = 0;
  var adresse = '';

  for (i = 0; i < pointsBassin.length; i += 1) {
    voisins = 0;

    for (j = 0; j < pointsBassin.length; j += 1) {
      if (i === j) {
        continue;
      }

      if (calculerDistanceMetres_(pointsBassin[i], pointsBassin[j]) <= 140) {
        voisins += 1;
      }
    }

    distanceCentre = calculerDistanceMetres_(pointsBassin[i], centre);
    adresse = String(pointsBassin[i].adresse || '').toLowerCase();

    if (pointsBassin.length <= 2 || /parking|impasse/.test(adresse) || (voisins <= 1 && distanceCentre >= 85)) {
      appendices.push(pointsBassin[i]);
    }
  }

  return appendices;
}

function essayerRepartirBassinsEnDeuxGroupes_(contexteBassins, depot, tailles) {
  var bassins = (contexteBassins && contexteBassins.bassins) || [];
  var groupeJ1 = [];
  var groupeJ2 = [];
  var scoreJ1 = 0;
  var scoreJ2 = 0;
  var candidatJ1 = null;
  var candidatJ2 = null;
  var i = 0;

  if (bassins.length <= 1 || bassins.length >= ((contexteBassins && contexteBassins.pointsLength) || Number.POSITIVE_INFINITY)) {
    return null;
  }

  bassins = bassins.slice().sort(function(a, b) {
    var ecartTaille = b.points.length - a.points.length;

    if (ecartTaille !== 0) {
      return ecartTaille;
    }

    return calculerDistanceMetres_(depot, b.centre) - calculerDistanceMetres_(depot, a.centre);
  });

  for (i = 0; i < bassins.length; i += 1) {
    if (groupeJ1.length + bassins[i].points.length > tailles[0] && groupeJ2.length + bassins[i].points.length > tailles[1]) {
      return null;
    }

    if (groupeJ1.length + bassins[i].points.length > tailles[0]) {
      Array.prototype.push.apply(groupeJ2, bassins[i].points);
      continue;
    }

    if (groupeJ2.length + bassins[i].points.length > tailles[1]) {
      Array.prototype.push.apply(groupeJ1, bassins[i].points);
      continue;
    }

    candidatJ1 = groupeJ1.concat(bassins[i].points);
    candidatJ2 = groupeJ2.concat(bassins[i].points);
    scoreJ1 = evaluerRepartitionGeographique_(candidatJ1, groupeJ2, depot, contexteBassins);
    scoreJ2 = evaluerRepartitionGeographique_(groupeJ1, candidatJ2, depot, contexteBassins);

    if (scoreJ1 <= scoreJ2) {
      Array.prototype.push.apply(groupeJ1, bassins[i].points);
    } else {
      Array.prototype.push.apply(groupeJ2, bassins[i].points);
    }
  }

  return [groupeJ1, groupeJ2];
}

function compterPointsMemeBassinDansGroupe_(point, groupe, contexteBassins, pointExclu) {
  var clePoint = construireClePoint_(point);
  var bassin = contexteBassins && contexteBassins.bassinParPoint ? contexteBassins.bassinParPoint[clePoint] : null;
  var total = 0;
  var i = 0;

  if (bassin === null || bassin === undefined || !groupe || groupe.length === 0) {
    return 0;
  }

  for (i = 0; i < groupe.length; i += 1) {
    if (pointExclu && groupe[i] === pointExclu) {
      continue;
    }

    if (contexteBassins.bassinParPoint[construireClePoint_(groupe[i])] === bassin) {
      total += 1;
    }
  }

  return total;
}

function calculerPenaliteSeparationBassinsGroupes_(groupes, contexteBassins) {
  var repartition = {};
  var point = null;
  var clePoint = '';
  var bassin = null;
  var compteurs = null;
  var penalite = 0;
  var i = 0;
  var j = 0;

  if (!contexteBassins || !contexteBassins.bassinParPoint) {
    return 0;
  }

  for (i = 0; i < (groupes || []).length; i += 1) {
    for (j = 0; j < (groupes[i] || []).length; j += 1) {
      point = groupes[i][j];
      clePoint = construireClePoint_(point);
      bassin = contexteBassins.bassinParPoint[clePoint];

      if (bassin === null || bassin === undefined) {
        continue;
      }

      if (!repartition[bassin]) {
        repartition[bassin] = [];
      }

      repartition[bassin][i] = (repartition[bassin][i] || 0) + 1;
    }
  }

  Object.keys(repartition).forEach(function(cleBassin) {
    compteurs = repartition[cleBassin].filter(function(count) {
      return count > 0;
    });

    if (compteurs.length > 1) {
      penalite += Math.min.apply(null, compteurs);
    }
  });

  return penalite;
}

function construireContexteUrbainTournee_(points, startIndex, endIndex, durations, distances) {
  var contexte = construireContexteBassinsGroupes_(points.slice(startIndex + 1, endIndex));
  var indexParPoint = {};
  var bassins = [];
  var appendixIndexes = {};
  var proximityDetourArcs = construireArcsDetourProximite_(points, durations, distances, startIndex, endIndex);
  var i = 0;
  var j = 0;
  var clePoint = '';

  for (i = startIndex + 1; i < endIndex; i += 1) {
    indexParPoint[construireClePoint_(points[i])] = i;
  }

  for (i = 0; i < contexte.bassins.length; i += 1) {
    bassins[i] = {
      id: i,
      points: contexte.bassins[i].points.slice(),
      centre: contexte.bassins[i].centre,
      indexes: contexte.bassins[i].points.map(function(point) {
        return indexParPoint[construireClePoint_(point)];
      }).filter(function(indexPoint) {
        return indexPoint !== undefined;
      })
    };

    for (j = 0; j < contexte.bassins[i].appendices.length; j += 1) {
      clePoint = construireClePoint_(contexte.bassins[i].appendices[j]);
      if (indexParPoint[clePoint] !== undefined) {
        appendixIndexes[indexParPoint[clePoint]] = true;
      }
    }
  }

  return {
    bassins: bassins,
    bassinParPoint: contexte.bassinParPoint,
    appendixIndexes: appendixIndexes,
    proximityDetourArcs: proximityDetourArcs
  };
}

function construireRouteInitialeParBassins_(points, durations, distances, startIndex, endIndex, contexteUrbain) {
  var bassinsRestants = null;
  var route = [];
  var meilleurBassin = null;
  var ordreBassin = null;
  var coutBassin = 0;
  var meilleurCout = 0;
  var currentIndex = startIndex;
  var i = 0;

  if (!contexteUrbain || !contexteUrbain.bassins || contexteUrbain.bassins.length < 2) {
    return null;
  }

  bassinsRestants = contexteUrbain.bassins.slice();

  while (bassinsRestants.length > 0) {
    meilleurBassin = null;
    meilleurCout = Number.POSITIVE_INFINITY;

    for (i = 0; i < bassinsRestants.length; i += 1) {
      coutBassin = calculerCoutEntreeBassin_(currentIndex, bassinsRestants[i], durations, points);

      if (coutBassin < meilleurCout) {
        meilleurCout = coutBassin;
        meilleurBassin = bassinsRestants[i];
      }
    }

    if (!meilleurBassin) {
      return null;
    }

    ordreBassin = construireOrdreInterneBassin_(meilleurBassin, currentIndex, durations, distances, points);
    Array.prototype.push.apply(route, ordreBassin.route);
    currentIndex = ordreBassin.lastIndex;
    bassinsRestants = bassinsRestants.filter(function(bassin) {
      return bassin !== meilleurBassin;
    });
  }

  return route.length === (endIndex - startIndex - 1) ? route : null;
}

function calculerCoutEntreeBassin_(currentIndex, bassin, durations, points) {
  var meilleurCout = Number.POSITIVE_INFINITY;
  var cout = 0;
  var i = 0;

  for (i = 0; i < bassin.indexes.length; i += 1) {
    cout = safeCost_(durations[currentIndex][bassin.indexes[i]]) + calculerDistanceMetres_(points[currentIndex], points[bassin.indexes[i]]) * 0.05;

    if (cout < meilleurCout) {
      meilleurCout = cout;
    }
  }

  return meilleurCout;
}

function construireOrdreInterneBassin_(bassin, currentIndex, durations, distances, points) {
  var restants = bassin.indexes.slice();
  var route = [];
  var meilleurIndex = -1;
  var meilleurCout = 0;
  var cout = 0;
  var i = 0;

  while (restants.length > 0) {
    meilleurIndex = 0;
    meilleurCout = Number.POSITIVE_INFINITY;

    for (i = 0; i < restants.length; i += 1) {
      cout = safeCost_(durations[currentIndex][restants[i]]) + safeCost_(distances[currentIndex][restants[i]]) * 0.01 + calculerDistanceMetres_(points[restants[i]], bassin.centre) * 0.05;

      if (cout < meilleurCout) {
        meilleurCout = cout;
        meilleurIndex = i;
      }
    }

    currentIndex = restants.splice(meilleurIndex, 1)[0];
    route.push(currentIndex);
  }

  return {
    route: route,
    lastIndex: currentIndex
  };
}

function construireConfigAjustementTrajectoire_() {
  return {
    maxPasses: 4,
    minLegDistance: 120,
    minLegDuration: 35,
    maxSegments: 10,
    priorityDistanceFactor: 0.12,
    shortSegmentDistanceLimit: 350,
    shortSegmentDurationLimit: 110,
    shortSegmentBonus: 35,
    baseThreshold: 70,
    distanceStep: 320,
    durationStep: 70,
    distanceThresholdCap: 28,
    durationThresholdCap: 22,
    maxThreshold: 120,
    durationFactor: 0.45,
    distanceFactor: 0.24,
    minDurationThreshold: 105,
    minDistanceThreshold: 280,
    maxDurationThreshold: 210,
    maxDistanceThreshold: 550,
    closeDistanceThreshold: 45,
    closeDurationBonus: 40,
    closeDistanceBonus: 90,
    positionsThreshold: 2,
    positionsDurationBonus: 25,
    positionsDistanceBonus: 70,
    forceCorridorDistance: 50,
    forceCorridorPositions: 2,
    forceDurationBonus: 50,
    forceDistanceBonus: 120,
    protectedPrefixSegments: 1,
    protectedPrefixCorridorDistance: 28,
    protectedPrefixMaxPositionsGained: 1,
    protectedPrefixMaxDurationThreshold: 55,
    protectedPrefixMaxDistanceThreshold: 160,
    protectedPrefixPriorityPenalty: 40,
    streetContinuityPriorityBonus: 45,
    streetContinuityDurationBonus: 18,
    streetContinuityDistanceBonus: 45,
    streetContinuityDistanceTolerance: 18,
    streetProjectionInteriorDistance: 22,
    passedPointDistanceThreshold: 50,
    passedPointPriorityBonus: 120,
    passedPointDurationBonus: 85,
    passedPointDistanceBonus: 240,
    passedPointDestinationRadius: 170
  };
}

function ajusterOrdreParTrajectoireORS_(points, ordreInterne, matrix, profil, apiKey) {
  var config = construireConfigAjustementTrajectoire_();
  var route = ordreInterne.slice();
  var pairesLocales = construirePairesLocales_(points, matrix.durations, matrix.distances, 0, points.length - 1);
  var opportunitesInsertion = construireOpportunitesInsertion_(points, matrix.durations, 0, points.length - 1);
  var contexteUrbain = construireContexteUrbainTournee_(points, 0, points.length - 1, matrix.durations, matrix.distances);
  var scoreCourant = evaluerParcours_(route, matrix.durations, matrix.distances, 0, points.length - 1, pairesLocales, opportunitesInsertion, points, contexteUrbain);
  var cacheTrajets = {};
  var amelioration = true;
  var passage = 0;
  var parcours = null;
  var segments = null;
  var segment = null;
  var trajectoire = null;
  var meilleurCandidat = null;
  var candidatIndex = 0;
  var distanceTrajectoire = 0;
  var projectionTrajectoire = null;
  var projectionInterieure = false;
  var pointSauteTrajectoire = false;
  var candidateRoute = null;
  var candidateScore = null;
  var surcoutDuree = 0;
  var surcoutDistance = 0;
  var positionsGagnees = 0;
  var prioriteCandidat = 0;
  var continuiteRue = 0;
  var continuiteRueBrute = 0;
  var i = 0;
  var position = 0;

  while (amelioration && passage < config.maxPasses) {
    amelioration = false;
    parcours = [0].concat(route).concat([points.length - 1]);
    segments = construireSegmentsCandidatsTrajectoire_(parcours, matrix, config);

    for (i = 0; i < segments.length; i += 1) {
      segment = segments[i];
      trajectoire = recupererTrajetDetailleORS_(points[segment.fromIndex], points[segment.toIndex], profil, apiKey, cacheTrajets);
      meilleurCandidat = null;

      for (position = segment.segmentIndex + 2; position < parcours.length - 1; position += 1) {
        candidatIndex = parcours[position];
        projectionTrajectoire = analyserProjectionPointPolyline_(points[candidatIndex], trajectoire);
        distanceTrajectoire = projectionTrajectoire.distance;
        positionsGagnees = position - (segment.segmentIndex + 1);

        if (distanceTrajectoire > calculerSeuilTrajectoireMetres_(segment.legDistance, segment.legDuration, config)) {
          continue;
        }

        if (estCandidatDebutTropAgressif_(segment.segmentIndex, distanceTrajectoire, positionsGagnees, config)) {
          continue;
        }

        candidateRoute = deplacerPointDansOrdre_(route, candidatIndex, segment.segmentIndex);
        candidateScore = evaluerParcours_(candidateRoute, matrix.durations, matrix.distances, 0, points.length - 1, pairesLocales, opportunitesInsertion, points, contexteUrbain);
        surcoutDuree = candidateScore.duration - scoreCourant.duration;
        surcoutDistance = candidateScore.distance - scoreCourant.distance;
        continuiteRueBrute = evaluerContinuiteRueSegment_(points[candidatIndex], points[segment.fromIndex], points[segment.toIndex]);
        projectionInterieure = estProjectionTrajectoireInterieure_(projectionTrajectoire, config);
        pointSauteTrajectoire = estPointSauteSurTrajectoire_(projectionTrajectoire, distanceTrajectoire, continuiteRueBrute, points[candidatIndex], points[segment.toIndex], config);

        if (continuiteRueBrute > 0 && !projectionInterieure && !pointSauteTrajectoire && !estMeilleurScore_(candidateScore, scoreCourant)) {
          continue;
        }

        continuiteRue = projectionInterieure ? continuiteRueBrute : 0;
        prioriteCandidat = calculerPrioriteCandidatTrajectoire_(distanceTrajectoire, surcoutDuree, surcoutDistance, positionsGagnees, continuiteRue, pointSauteTrajectoire, projectionTrajectoire, segment, config);

        if (!estAjustementTrajectoireAcceptable_(scoreCourant, candidateScore, surcoutDuree, surcoutDistance, distanceTrajectoire, segment.legDistance, segment.legDuration, positionsGagnees, continuiteRue, pointSauteTrajectoire, segment.segmentIndex, config)) {
          continue;
        }

        if (!meilleurCandidat || estCandidatTrajectoireMeilleur_({
          distanceTrajectoire: distanceTrajectoire,
          surcoutDuree: surcoutDuree,
          surcoutDistance: surcoutDistance,
          positionsGagnees: positionsGagnees,
          priorite: prioriteCandidat,
          continuiteRue: continuiteRue,
          pointSauteTrajectoire: pointSauteTrajectoire,
          distanceDepuisOrigineTrajectoire: projectionTrajectoire ? projectionTrajectoire.distanceToStart : Number.POSITIVE_INFINITY
        }, meilleurCandidat, config)) {
          meilleurCandidat = {
            pointIndex: candidatIndex,
            distanceTrajectoire: distanceTrajectoire,
            route: candidateRoute,
            score: candidateScore,
            surcoutDuree: surcoutDuree,
            surcoutDistance: surcoutDistance,
            positionsGagnees: positionsGagnees,
            priorite: prioriteCandidat,
            continuiteRue: continuiteRue,
            pointSauteTrajectoire: pointSauteTrajectoire,
            distanceDepuisOrigineTrajectoire: projectionTrajectoire ? projectionTrajectoire.distanceToStart : Number.POSITIVE_INFINITY
          };
        }
      }

      if (meilleurCandidat) {
        route = meilleurCandidat.route;
        scoreCourant = meilleurCandidat.score;
        amelioration = true;
        break;
      }
    }

    passage += 1;
  }

  return route;
}

function construireSegmentsCandidatsTrajectoire_(parcours, matrix, config) {
  var segments = [];
  var fromIndex = 0;
  var toIndex = 0;
  var legDistance = 0;
  var legDuration = 0;
  var priority = 0;
  var i = 0;

  for (i = 0; i < parcours.length - 2; i += 1) {
    fromIndex = parcours[i];
    toIndex = parcours[i + 1];
    legDistance = safeCost_(matrix.distances[fromIndex][toIndex]);
    legDuration = safeCost_(matrix.durations[fromIndex][toIndex]);

    if (!isFinite(legDistance) || !isFinite(legDuration)) {
      continue;
    }

    if (legDistance < config.minLegDistance && legDuration < config.minLegDuration) {
      continue;
    }

    priority = legDuration + Math.min(legDistance, 2500) * config.priorityDistanceFactor;

    if (
      config.shortSegmentBonus > 0
      && legDistance <= config.shortSegmentDistanceLimit
      && legDuration <= config.shortSegmentDurationLimit
    ) {
      priority += config.shortSegmentBonus;
    }

    segments.push({
      segmentIndex: i,
      fromIndex: fromIndex,
      toIndex: toIndex,
      legDistance: legDistance,
      legDuration: legDuration,
      priority: priority
    });
  }

  return segments.sort(function(a, b) {
    return b.priority - a.priority;
  }).slice(0, config.maxSegments);
}

function calculerSeuilTrajectoireMetres_(legDistance, legDuration, config) {
  var seuil = config.baseThreshold;

  seuil += Math.min(config.distanceThresholdCap, Math.floor(legDistance / config.distanceStep));
  seuil += Math.min(config.durationThresholdCap, Math.floor(legDuration / config.durationStep));
  return Math.min(seuil, config.maxThreshold);
}

function recupererTrajetDetailleORS_(pointA, pointB, profil, apiKey, cacheTrajets) {
  var cacheKey = [profil, pointA.longitude, pointA.latitude, pointB.longitude, pointB.latitude].join('|');
  var url = 'https://api.openrouteservice.org/v2/directions/' + encodeURIComponent(profil) + '/geojson';
  var payload = {
    coordinates: [
      [pointA.longitude, pointA.latitude],
      [pointB.longitude, pointB.latitude]
    ]
  };
  var response = null;
  var status = 0;
  var json = null;
  var coordinates = null;

  if (cacheTrajets[cacheKey]) {
    return cacheTrajets[cacheKey];
  }

  try {
    response = UrlFetchApp.fetch(url, {
      method: 'post',
      contentType: 'application/json',
      headers: {
        Authorization: apiKey
      },
      payload: JSON.stringify(payload),
      muteHttpExceptions: true
    });
    status = response.getResponseCode();

    if (status >= 200 && status < 300) {
      json = JSON.parse(response.getContentText());
      coordinates = json && json.features && json.features[0] && json.features[0].geometry
        ? json.features[0].geometry.coordinates
        : null;

      if (coordinates && coordinates.length > 0) {
        cacheTrajets[cacheKey] = coordinates.map(function(coordinate) {
          return {
            longitude: coordinate[0],
            latitude: coordinate[1]
          };
        });
        return cacheTrajets[cacheKey];
      }
    }
  } catch (error) {
  }

  cacheTrajets[cacheKey] = [
    {
      longitude: pointA.longitude,
      latitude: pointA.latitude
    },
    {
      longitude: pointB.longitude,
      latitude: pointB.latitude
    }
  ];

  return cacheTrajets[cacheKey];
}

function calculerPrioriteCandidatTrajectoire_(distanceTrajectoire, surcoutDuree, surcoutDistance, positionsGagnees, continuiteRue, pointSauteTrajectoire, projectionTrajectoire, segment, config) {
  var priorite = 0;

  priorite += Math.max(0, config.maxThreshold - distanceTrajectoire) * 1.8;
  priorite += positionsGagnees * 35;
  priorite += Math.min(40, Math.round(segment.legDuration / 25));
  priorite += continuiteRue * config.streetContinuityPriorityBonus;
  if (pointSauteTrajectoire) {
    priorite += config.passedPointPriorityBonus;
    priorite += Math.min(40, Math.round((projectionTrajectoire ? projectionTrajectoire.distanceToEnd : 0) / 12));
  }
  priorite -= Math.max(0, surcoutDuree) * 0.35;
  priorite -= Math.max(0, surcoutDistance) * 0.05;

  if (distanceTrajectoire <= config.forceCorridorDistance && positionsGagnees >= config.forceCorridorPositions) {
    priorite += 60;
  }

  if (segment.segmentIndex < config.protectedPrefixSegments) {
    priorite -= positionsGagnees * config.protectedPrefixPriorityPenalty;
  }

  return priorite;
}

function estCandidatDebutTropAgressif_(segmentIndex, distanceTrajectoire, positionsGagnees, config) {
  if (config.protectedPrefixSegments <= 0 || segmentIndex >= config.protectedPrefixSegments) {
    return false;
  }

  if (distanceTrajectoire > config.protectedPrefixCorridorDistance) {
    return true;
  }

  return positionsGagnees > config.protectedPrefixMaxPositionsGained;
}

function estAjustementTrajectoireAcceptable_(scoreCourant, scoreCandidat, surcoutDuree, surcoutDistance, distanceTrajectoire, legDistance, legDuration, positionsGagnees, continuiteRue, pointSauteTrajectoire, segmentIndex, config) {
  var seuilDuree = Math.min(config.maxDurationThreshold, Math.max(config.minDurationThreshold, Math.round(legDuration * config.durationFactor)));
  var seuilDistance = Math.min(config.maxDistanceThreshold, Math.max(config.minDistanceThreshold, Math.round(legDistance * config.distanceFactor)));

  if (estMeilleurScore_(scoreCandidat, scoreCourant)) {
    return true;
  }

  if (distanceTrajectoire <= config.closeDistanceThreshold) {
    seuilDuree += config.closeDurationBonus;
    seuilDistance += config.closeDistanceBonus;
  }

  if (positionsGagnees >= config.positionsThreshold) {
    seuilDuree += config.positionsDurationBonus;
    seuilDistance += config.positionsDistanceBonus;
  }

  if (distanceTrajectoire <= config.forceCorridorDistance && positionsGagnees >= config.forceCorridorPositions) {
    seuilDuree += config.forceDurationBonus;
    seuilDistance += config.forceDistanceBonus;
  }

  if (continuiteRue > 0) {
    seuilDuree += continuiteRue * config.streetContinuityDurationBonus;
    seuilDistance += continuiteRue * config.streetContinuityDistanceBonus;
  }

  if (pointSauteTrajectoire) {
    seuilDuree += config.passedPointDurationBonus;
    seuilDistance += config.passedPointDistanceBonus;
  }

  if (segmentIndex < config.protectedPrefixSegments) {
    seuilDuree = Math.min(seuilDuree, config.protectedPrefixMaxDurationThreshold);
    seuilDistance = Math.min(seuilDistance, config.protectedPrefixMaxDistanceThreshold);
  }

  return surcoutDuree <= seuilDuree && surcoutDistance <= seuilDistance;
}

function estCandidatTrajectoireMeilleur_(candidat, meilleurCandidat, config) {
  if (candidat.pointSauteTrajectoire !== meilleurCandidat.pointSauteTrajectoire) {
    return candidat.pointSauteTrajectoire;
  }

  if (
    candidat.pointSauteTrajectoire
    && meilleurCandidat.pointSauteTrajectoire
    && Math.abs(candidat.distanceDepuisOrigineTrajectoire - meilleurCandidat.distanceDepuisOrigineTrajectoire) > 8
  ) {
    return candidat.distanceDepuisOrigineTrajectoire < meilleurCandidat.distanceDepuisOrigineTrajectoire;
  }

  if (
    candidat.continuiteRue !== meilleurCandidat.continuiteRue
    && Math.abs(candidat.distanceTrajectoire - meilleurCandidat.distanceTrajectoire) <= config.streetContinuityDistanceTolerance
  ) {
    return candidat.continuiteRue > meilleurCandidat.continuiteRue;
  }

  if (candidat.priorite > meilleurCandidat.priorite + 2) {
    return true;
  }

  if (meilleurCandidat.priorite > candidat.priorite + 2) {
    return false;
  }

  if (candidat.distanceTrajectoire + 5 < meilleurCandidat.distanceTrajectoire) {
    return true;
  }

  if (meilleurCandidat.distanceTrajectoire + 5 < candidat.distanceTrajectoire) {
    return false;
  }

  if (candidat.continuiteRue !== meilleurCandidat.continuiteRue) {
    return candidat.continuiteRue > meilleurCandidat.continuiteRue;
  }

  if (candidat.positionsGagnees !== meilleurCandidat.positionsGagnees) {
    return candidat.positionsGagnees > meilleurCandidat.positionsGagnees;
  }

  if (candidat.surcoutDuree !== meilleurCandidat.surcoutDuree) {
    return candidat.surcoutDuree < meilleurCandidat.surcoutDuree;
  }

  return candidat.surcoutDistance < meilleurCandidat.surcoutDistance;
}

function deplacerPointDansOrdre_(route, pointIndex, positionInsertion) {
  var nouvelleRoute = route.filter(function(indexPoint) {
    return indexPoint !== pointIndex;
  });

  nouvelleRoute.splice(Math.max(0, Math.min(positionInsertion, nouvelleRoute.length)), 0, pointIndex);
  return nouvelleRoute;
}

function distancePointPolylineMetres_(point, polyline) {
  return analyserProjectionPointPolyline_(point, polyline).distance;
}

function distancePointSegmentMetres_(point, segmentStart, segmentEnd) {
  return analyserProjectionPointSegmentMetres_(point, segmentStart, segmentEnd).distance;
}

function analyserProjectionPointPolyline_(point, polyline) {
  var meilleureProjection = null;
  var longueurTotale = 0;
  var longueurAvantSegment = 0;
  var longueurSegment = 0;
  var analyseSegment = null;
  var i = 0;

  if (!polyline || polyline.length === 0) {
    return {
      distance: Number.POSITIVE_INFINITY,
      distanceToStart: Number.POSITIVE_INFINITY,
      distanceToEnd: Number.POSITIVE_INFINITY,
      totalLength: 0
    };
  }

  if (polyline.length === 1) {
    longueurTotale = calculerDistanceMetres_(point, polyline[0]);
    return {
      distance: longueurTotale,
      distanceToStart: 0,
      distanceToEnd: 0,
      totalLength: 0
    };
  }

  for (i = 0; i < polyline.length - 1; i += 1) {
    longueurTotale += calculerDistanceMetres_(polyline[i], polyline[i + 1]);
  }

  for (i = 0; i < polyline.length - 1; i += 1) {
    longueurSegment = calculerDistanceMetres_(polyline[i], polyline[i + 1]);
    analyseSegment = analyserProjectionPointSegmentMetres_(point, polyline[i], polyline[i + 1]);

    if (!meilleureProjection || analyseSegment.distance < meilleureProjection.distance) {
      meilleureProjection = {
        distance: analyseSegment.distance,
        distanceToStart: longueurAvantSegment + longueurSegment * analyseSegment.projection,
        distanceToEnd: 0,
        totalLength: longueurTotale
      };
    }

    longueurAvantSegment += longueurSegment;
  }

  if (!meilleureProjection) {
    return {
      distance: Number.POSITIVE_INFINITY,
      distanceToStart: Number.POSITIVE_INFINITY,
      distanceToEnd: Number.POSITIVE_INFINITY,
      totalLength: longueurTotale
    };
  }

  meilleureProjection.distanceToEnd = Math.max(0, longueurTotale - meilleureProjection.distanceToStart);
  return meilleureProjection;
}

function analyserProjectionPointSegmentMetres_(point, segmentStart, segmentEnd) {
  var referenceLatitude = (point.latitude + segmentStart.latitude + segmentEnd.latitude) / 3;
  var pointXY = convertirCoordonneesEnMetres_(point.latitude, point.longitude, referenceLatitude);
  var startXY = convertirCoordonneesEnMetres_(segmentStart.latitude, segmentStart.longitude, referenceLatitude);
  var endXY = convertirCoordonneesEnMetres_(segmentEnd.latitude, segmentEnd.longitude, referenceLatitude);
  var dx = endXY.x - startXY.x;
  var dy = endXY.y - startXY.y;
  var projection = 0;
  var projX = 0;
  var projY = 0;

  if (dx === 0 && dy === 0) {
    return {
      distance: Math.sqrt(Math.pow(pointXY.x - startXY.x, 2) + Math.pow(pointXY.y - startXY.y, 2)),
      projection: 0
    };
  }

  projection = ((pointXY.x - startXY.x) * dx + (pointXY.y - startXY.y) * dy) / (dx * dx + dy * dy);
  projection = Math.max(0, Math.min(1, projection));
  projX = startXY.x + projection * dx;
  projY = startXY.y + projection * dy;

  return {
    distance: Math.sqrt(Math.pow(pointXY.x - projX, 2) + Math.pow(pointXY.y - projY, 2)),
    projection: projection
  };
}

function estProjectionTrajectoireInterieure_(projectionTrajectoire, config) {
  var marge = 0;

  if (!projectionTrajectoire || !isFinite(projectionTrajectoire.distance)) {
    return false;
  }

  marge = Math.min(config.streetProjectionInteriorDistance, Math.max(10, projectionTrajectoire.totalLength * 0.2));
  return projectionTrajectoire.distanceToStart >= marge && projectionTrajectoire.distanceToEnd >= marge;
}

function estPointSauteSurTrajectoire_(projectionTrajectoire, distanceTrajectoire, continuiteRue, pointCandidat, pointDestination, config) {
  if (!projectionTrajectoire || !pointCandidat || !pointDestination) {
    return false;
  }

  if (!estProjectionTrajectoireInterieure_(projectionTrajectoire, config)) {
    return false;
  }

  if (!isFinite(distanceTrajectoire) || distanceTrajectoire > config.passedPointDistanceThreshold) {
    return false;
  }

  if (continuiteRue > 0) {
    return true;
  }

  return calculerDistanceMetres_(pointCandidat, pointDestination) <= config.passedPointDestinationRadius;
}

function evaluerContinuiteRueSegment_(pointCandidat, pointFrom, pointTo) {
  var rueCandidate = canoniserNomRue_(pointCandidat && pointCandidat.adresse);
  var rueFrom = canoniserNomRue_(pointFrom && pointFrom.adresse);
  var rueTo = canoniserNomRue_(pointTo && pointTo.adresse);
  var score = 0;

  if (!rueCandidate) {
    return 0;
  }

  if (rueFrom && rueCandidate === rueFrom) {
    score += 1;
  }

  if (rueTo && rueCandidate === rueTo) {
    score += 1;
  }

  return score;
}

function canoniserNomRue_(adresse) {
  var texte = String(adresse || '').trim().toLowerCase();

  if (!texte) {
    return '';
  }

  texte = texte
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/\([^)]*\)/g, '')
    .split(',').pop()
    .replace(/^\s*face\s+au\s+n[°o]?\s*\d+\s*/i, '')
    .replace(/^\s*en\s+face\s+du\s+n[°o]?\s*\d+\s*/i, '')
    .replace(/^\s*\d+[a-z]?\s*(bis|ter|quater)?\s+/i, '')
    .replace(/\bav\.?\b/g, 'avenue')
    .replace(/\bbd\.?\b/g, 'boulevard')
    .replace(/\bfg\.?\b/g, 'faubourg')
    .replace(/\bpl\.?\b/g, 'place')
    .replace(/\brte\.?\b/g, 'route')
    .replace(/\bimp\.?\b/g, 'impasse')
    .replace(/\bche\.?\b/g, 'chemin')
    .replace(/\bsq\.?\b/g, 'square')
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  return texte.length >= 4 ? texte : '';
}

function convertirCoordonneesEnMetres_(latitude, longitude, referenceLatitude) {
  var rayonTerre = 6371000;
  var refLatRad = degreesToRadians_(referenceLatitude);

  return {
    x: degreesToRadians_(longitude) * rayonTerre * Math.cos(refLatRad),
    y: degreesToRadians_(latitude) * rayonTerre
  };
}

function calculerOrdreOptimise_(points, durations, distances) {
  var totalPoints = durations.length;
  if (totalPoints < 3) {
    return [];
  }

  var startIndex = 0;
  var endIndex = totalPoints - 1;
  var candidats = [];
  var pairesLocales = null;
  var opportunitesInsertion = null;
  var graines = null;
  var meilleureRoute = [];
  var meilleurScore = null;
  var contexteUrbain = null;
  var routeParBassins = null;
  var route = null;
  var score = null;
  var perturbee = null;
  var maxPerturbations = 3;
  var deadlineILS = Date.now() + 20000;
  var iterILS = 0;
  var i = 0;

  for (i = 1; i < endIndex; i += 1) {
    candidats.push(i);
  }

  if (candidats.length === 1) {
    return candidats.slice();
  }

  pairesLocales = construirePairesLocales_(points, durations, distances, startIndex, endIndex);
  opportunitesInsertion = construireOpportunitesInsertion_(points, durations, startIndex, endIndex);
  contexteUrbain = construireContexteUrbainTournee_(points, startIndex, endIndex, durations, distances);
  graines = genererGrainesOptimisation_(candidats, durations, startIndex, endIndex);

  routeParBassins = construireRouteInitialeParBassins_(points, durations, distances, startIndex, endIndex, contexteUrbain);
  if (routeParBassins && routeParBassins.length === candidats.length) {
    routeParBassins = ameliorerParRechercheLocale_(routeParBassins, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    meilleurScore = evaluerParcours_(routeParBassins, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    meilleureRoute = routeParBassins.slice();
  }

  for (i = 0; i < graines.length; i += 1) {
    route = construireRouteParInsertion_(candidats, graines[i], durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    route = ameliorerParRechercheLocale_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    score = evaluerParcours_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

    if (estMeilleurScore_(score, meilleurScore)) {
      meilleureRoute = route;
      meilleurScore = score;
    }
  }

  iterILS = 0;
  while (iterILS < maxPerturbations && meilleureRoute.length >= 4 && Date.now() < deadlineILS) {
    perturbee = perturberRoute_(meilleureRoute);
    perturbee = ameliorerParRechercheLocale_(perturbee, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    score = evaluerParcours_(perturbee, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

    if (estMeilleurScore_(score, meilleurScore)) {
      meilleureRoute = perturbee;
      meilleurScore = score;
    }

    iterILS += 1;
  }

  return meilleureRoute;
}

function perturberRoute_(route) {
  var perturbee = route.slice();
  var n = perturbee.length;
  var i1 = 0;
  var i2 = 0;
  var i3 = 0;
  var i4 = 0;
  var tampon = 0;

  if (n < 4) {
    return perturbee;
  }

  i1 = Math.floor(Math.random() * n);
  i2 = Math.floor(Math.random() * (n - 1));
  if (i2 >= i1) {
    i2 += 1;
  }

  tampon = perturbee[i1];
  perturbee[i1] = perturbee[i2];
  perturbee[i2] = tampon;

  i3 = Math.floor(Math.random() * n);
  i4 = Math.floor(Math.random() * (n - 1));
  if (i4 >= i3) {
    i4 += 1;
  }

  tampon = perturbee[i3];
  perturbee[i3] = perturbee[i4];
  perturbee[i4] = tampon;

  return perturbee;
}

function genererGrainesOptimisation_(candidats, durations, startIndex, endIndex) {
  var sortedByStart = candidats.slice().sort(function(a, b) {
    return safeCost_(durations[startIndex][a]) - safeCost_(durations[startIndex][b]);
  });
  var sortedByEnd = candidats.slice().sort(function(a, b) {
    return safeCost_(durations[a][endIndex]) - safeCost_(durations[b][endIndex]);
  });
  var positions = [];
  var graines = [];
  var maxGraines = Math.min(6, candidats.length);
  var i = 0;

  if (candidats.length === 0) {
    return [];
  }

  positions = [0, Math.floor((sortedByStart.length - 1) / 3), Math.floor((sortedByStart.length - 1) / 2), sortedByStart.length - 1];

  for (i = 0; i < positions.length; i += 1) {
    ajouterUnique_(graines, sortedByStart[positions[i]]);
  }

  ajouterUnique_(graines, sortedByEnd[0]);
  ajouterUnique_(graines, sortedByEnd[sortedByEnd.length - 1]);

  return graines.slice(0, maxGraines);
}

function construireRouteParInsertion_(candidats, graine, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain) {
  var route = [graine];
  var restants = candidats.filter(function(candidat) {
    return candidat !== graine;
  });
  var meilleurCout = null;
  var meilleureRoute = route.slice();
  var candidateRoute = null;
  var candidateScore = null;
  var meilleurIndexRestant = -1;
  var seuilVoisinage = 300;
  var maxExtensions = 2;
  var pointInsere = 0;
  var meilleurVoisinIndex = -1;
  var meilleureRouteVoisin = null;
  var meilleurScoreVoisin = null;
  var scoreActuel = null;
  var extensionFaite = false;
  var nbExt = 0;
  var distVoisin = 0;
  var i = 0;
  var j = 0;
  var position = 0;

  while (restants.length > 0) {
    meilleurCout = null;
    meilleureRoute = route.slice();
    meilleurIndexRestant = -1;

    for (i = 0; i < restants.length; i += 1) {
      for (position = 0; position <= route.length; position += 1) {
        candidateRoute = route.slice();
        candidateRoute.splice(position, 0, restants[i]);
        candidateScore = evaluerParcours_(candidateRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

        if (estMeilleurScore_(candidateScore, meilleurCout)) {
          meilleurCout = candidateScore;
          meilleureRoute = candidateRoute;
          meilleurIndexRestant = i;
        }
      }
    }

    pointInsere = restants[meilleurIndexRestant];
    route = meilleureRoute;
    restants.splice(meilleurIndexRestant, 1);

    nbExt = 0;
    extensionFaite = true;

    while (extensionFaite && nbExt < maxExtensions && restants.length > 0) {
      extensionFaite = false;
      meilleurVoisinIndex = -1;
      meilleureRouteVoisin = route;
      scoreActuel = evaluerParcours_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
      meilleurScoreVoisin = scoreActuel;

      for (j = 0; j < restants.length; j += 1) {
        distVoisin = calculerDistanceMetres_(points[restants[j]], points[pointInsere]);
        if (distVoisin > seuilVoisinage) {
          continue;
        }

        for (position = 0; position <= route.length; position += 1) {
          candidateRoute = route.slice();
          candidateRoute.splice(position, 0, restants[j]);
          candidateScore = evaluerParcours_(candidateRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

          if (estMeilleurScore_(candidateScore, meilleurScoreVoisin)) {
            meilleurScoreVoisin = candidateScore;
            meilleureRouteVoisin = candidateRoute;
            meilleurVoisinIndex = j;
          }
        }
      }

      if (meilleurVoisinIndex >= 0) {
        pointInsere = restants[meilleurVoisinIndex];
        route = meilleureRouteVoisin;
        restants.splice(meilleurVoisinIndex, 1);
        extensionFaite = true;
        nbExt += 1;
      }
    }
  }

  return route;
}

function ameliorerParRechercheLocale_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain) {
  var meilleureRoute = route.slice();
  var meilleurScore = evaluerParcours_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
  var tentative = null;
  var iteration = 0;
  var amelioration = true;

  while (amelioration && iteration < 5) {
    amelioration = false;

    tentative = ameliorerParRegroupementVoisinage_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    if (estMeilleurScore_(tentative.score, meilleurScore)) {
      meilleureRoute = tentative.route;
      meilleurScore = tentative.score;
      amelioration = true;
    }

    tentative = ameliorerParOrOpt_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    if (estMeilleurScore_(tentative.score, meilleurScore)) {
      meilleureRoute = tentative.route;
      meilleurScore = tentative.score;
      amelioration = true;
    }

    tentative = ameliorerParRelocate_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    if (estMeilleurScore_(tentative.score, meilleurScore)) {
      meilleureRoute = tentative.route;
      meilleurScore = tentative.score;
      amelioration = true;
    }

    tentative = ameliorerParSwap_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    if (estMeilleurScore_(tentative.score, meilleurScore)) {
      meilleureRoute = tentative.route;
      meilleurScore = tentative.score;
      amelioration = true;
    }

    tentative = ameliorerParFenetreLocale_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    if (estMeilleurScore_(tentative.score, meilleurScore)) {
      meilleureRoute = tentative.route;
      meilleurScore = tentative.score;
      amelioration = true;
    }

    tentative = ameliorerPar2Opt_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
    if (estMeilleurScore_(tentative.score, meilleurScore)) {
      meilleureRoute = tentative.route;
      meilleurScore = tentative.score;
      amelioration = true;
    }

    iteration += 1;
  }

  return meilleureRoute;
}

function ameliorerParRelocate_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain) {
  var meilleureRoute = route.slice();
  var meilleurScore = evaluerParcours_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
  var candidateRoute = null;
  var candidateScore = null;
  var noeud = null;
  var i = 0;
  var position = 0;
  var nouvellePosition = 0;

  for (i = 0; i < route.length; i += 1) {
    for (position = 0; position <= route.length; position += 1) {
      if (position === i || position === i + 1) {
        continue;
      }

      candidateRoute = route.slice();
      noeud = candidateRoute.splice(i, 1)[0];
      nouvellePosition = position;

      if (position > i) {
        nouvellePosition -= 1;
      }

      candidateRoute.splice(nouvellePosition, 0, noeud);
      candidateScore = evaluerParcours_(candidateRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

      if (estMeilleurScore_(candidateScore, meilleurScore)) {
        meilleureRoute = candidateRoute;
        meilleurScore = candidateScore;
      }
    }
  }

  return {
    route: meilleureRoute,
    score: meilleurScore
  };
}

function ameliorerParSwap_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain) {
  var meilleureRoute = route.slice();
  var meilleurScore = evaluerParcours_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
  var candidateRoute = null;
  var candidateScore = null;
  var tampon = null;
  var i = 0;
  var j = 0;

  for (i = 0; i < route.length - 1; i += 1) {
    for (j = i + 1; j < route.length; j += 1) {
      candidateRoute = route.slice();
      tampon = candidateRoute[i];
      candidateRoute[i] = candidateRoute[j];
      candidateRoute[j] = tampon;
      candidateScore = evaluerParcours_(candidateRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

      if (estMeilleurScore_(candidateScore, meilleurScore)) {
        meilleureRoute = candidateRoute;
        meilleurScore = candidateScore;
      }
    }
  }

  return {
    route: meilleureRoute,
    score: meilleurScore
  };
}

function ameliorerPar2Opt_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain) {
  var meilleureRoute = route.slice();
  var meilleurScore = evaluerParcours_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
  var candidateRoute = null;
  var candidateScore = null;
  var i = 0;
  var k = 0;

  if (route.length < 3) {
    return {
      route: meilleureRoute,
      score: meilleurScore
    };
  }

  for (i = 0; i < route.length - 1; i += 1) {
    for (k = i + 1; k < route.length; k += 1) {
      candidateRoute = twoOptSwap_(route, i, k);
      candidateScore = evaluerParcours_(candidateRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

      if (estMeilleurScore_(candidateScore, meilleurScore)) {
        meilleureRoute = candidateRoute;
        meilleurScore = candidateScore;
      }
    }
  }

  return {
    route: meilleureRoute,
    score: meilleurScore
  };
}

function ameliorerParOrOpt_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain) {
  var meilleureRoute = route.slice();
  var meilleurScore = evaluerParcours_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
  var candidateRoute = null;
  var candidateScore = null;
  var tailleChaineMax = Math.min(3, Math.floor(route.length / 2));
  var tailleChaine = 0;
  var i = 0;
  var position = 0;
  var chaine = null;
  var routeSansChaine = null;

  if (route.length < 4) {
    return {
      route: meilleureRoute,
      score: meilleurScore
    };
  }

  for (tailleChaine = 2; tailleChaine <= tailleChaineMax; tailleChaine += 1) {
    for (i = 0; i <= route.length - tailleChaine; i += 1) {
      chaine = route.slice(i, i + tailleChaine);
      routeSansChaine = route.slice(0, i).concat(route.slice(i + tailleChaine));

      for (position = 0; position <= routeSansChaine.length; position += 1) {
        if (position === i) {
          continue;
        }

        candidateRoute = routeSansChaine.slice();
        candidateRoute.splice.apply(candidateRoute, [position, 0].concat(chaine));
        candidateScore = evaluerParcours_(candidateRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

        if (estMeilleurScore_(candidateScore, meilleurScore)) {
          meilleureRoute = candidateRoute;
          meilleurScore = candidateScore;
        }
      }
    }
  }

  return {
    route: meilleureRoute,
    score: meilleurScore
  };
}

function ameliorerParRegroupementVoisinage_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain) {
  var meilleureRoute = route.slice();
  var meilleurScore = evaluerParcours_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
  var seuilVoisinage = 400;
  var candidateRoute = null;
  var candidateScore = null;
  var distGeo = 0;
  var noeud = 0;
  var cible = 0;
  var i = 0;
  var j = 0;
  var p = 0;
  var positionsTest = null;

  if (route.length < 4) {
    return {
      route: meilleureRoute,
      score: meilleurScore
    };
  }

  for (i = 0; i < route.length; i += 1) {
    for (j = 0; j < route.length; j += 1) {
      if (Math.abs(i - j) <= 2) {
        continue;
      }

      distGeo = calculerDistanceMetres_(points[route[i]], points[route[j]]);
      if (distGeo > seuilVoisinage) {
        continue;
      }

      noeud = route[j];
      cible = (j < i) ? i - 1 : i;
      positionsTest = [cible, cible + 1];
      if (cible > 0) {
        positionsTest.push(cible - 1);
      }

      for (p = 0; p < positionsTest.length; p += 1) {
        if (positionsTest[p] < 0 || positionsTest[p] > route.length - 1) {
          continue;
        }

        candidateRoute = route.slice();
        candidateRoute.splice(j, 1);
        candidateRoute.splice(positionsTest[p], 0, noeud);
        candidateScore = evaluerParcours_(candidateRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

        if (estMeilleurScore_(candidateScore, meilleurScore)) {
          meilleureRoute = candidateRoute;
          meilleurScore = candidateScore;
        }
      }
    }
  }

  return {
    route: meilleureRoute,
    score: meilleurScore
  };
}

function ameliorerParFenetreLocale_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain) {
  var meilleureRoute = route.slice();
  var meilleurScore = evaluerParcours_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
  var tailles = route.length >= 4 ? [4, 3] : [3];
  var taille = 0;
  var debut = 0;
  var fenetre = null;
  var permutations = null;
  var permutation = null;
  var candidateRoute = null;
  var candidateScore = null;
  var i = 0;
  var j = 0;

  if (route.length < 3) {
    return {
      route: meilleureRoute,
      score: meilleurScore
    };
  }

  for (i = 0; i < tailles.length; i += 1) {
    taille = tailles[i];

    if (route.length < taille) {
      continue;
    }

    for (debut = 0; debut <= route.length - taille; debut += 1) {
      fenetre = route.slice(debut, debut + taille);
      permutations = listerPermutationsTableau_(fenetre);

      for (j = 0; j < permutations.length; j += 1) {
        permutation = permutations[j];

        if (sontTableauxEgaux_(permutation, fenetre)) {
          continue;
        }

        candidateRoute = route.slice();
        candidateRoute.splice.apply(candidateRoute, [debut, taille].concat(permutation));
        candidateScore = evaluerParcours_(candidateRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

        if (estMeilleurScore_(candidateScore, meilleurScore)) {
          meilleureRoute = candidateRoute;
          meilleurScore = candidateScore;
        }
      }
    }
  }

  return {
    route: meilleureRoute,
    score: meilleurScore
  };
}

function listerPermutationsTableau_(elements) {
  var resultats = [];
  var courant = [];
  var utilises = [];

  function explorer() {
    var i = 0;

    if (courant.length === elements.length) {
      resultats.push(courant.slice());
      return;
    }

    for (i = 0; i < elements.length; i += 1) {
      if (utilises[i]) {
        continue;
      }

      utilises[i] = true;
      courant.push(elements[i]);
      explorer();
      courant.pop();
      utilises[i] = false;
    }
  }

  explorer();
  return resultats;
}

function sontTableauxEgaux_(a, b) {
  var i = 0;

  if (!a || !b || a.length !== b.length) {
    return false;
  }

  for (i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }

  return true;
}

function reordonnerMicroSecteursContigus_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain) {
  var meilleureRoute = route.slice();
  var meilleurScore = evaluerParcours_(meilleureRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);
  var blocs = [];
  var bloc = null;
  var resultatBloc = null;
  var sousBloc = null;
  var permutationSource = null;
  var permutations = null;
  var permutation = null;
  var candidateRoute = null;
  var candidateScore = null;
  var iteration = 0;
  var amelioration = true;
  var i = 0;
  var j = 0;

  if (!route || route.length < 2 || !contexteUrbain) {
    return route ? route.slice() : [];
  }

  while (amelioration && iteration < 2) {
    amelioration = false;
    blocs = listerBlocsMicroSecteursContigus_(meilleureRoute, points, contexteUrbain);

    for (i = 0; i < blocs.length; i += 1) {
      bloc = blocs[i];

      if ((bloc.end - bloc.start + 1) <= 6) {
        resultatBloc = testerReecritureBlocMicroSecteur_(meilleureRoute, bloc.start, bloc.end, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain, meilleurScore);
        if (resultatBloc) {
          meilleureRoute = resultatBloc.route;
          meilleurScore = resultatBloc.score;
          amelioration = true;
          break;
        }

        continue;
      }

      for (j = bloc.start; j <= bloc.end - 5; j += 1) {
        sousBloc = {
          start: j,
          end: Math.min(bloc.end, j + 5)
        };
        permutationSource = meilleureRoute.slice(sousBloc.start, sousBloc.end + 1);
        permutations = listerPermutationsTableau_(permutationSource);

        for (var k = 0; k < permutations.length; k += 1) {
          permutation = permutations[k];

          if (sontTableauxEgaux_(permutation, permutationSource)) {
            continue;
          }

          candidateRoute = meilleureRoute.slice();
          candidateRoute.splice.apply(candidateRoute, [sousBloc.start, permutationSource.length].concat(permutation));
          candidateScore = evaluerParcours_(candidateRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

          if (estMeilleurScore_(candidateScore, meilleurScore)) {
            meilleureRoute = candidateRoute;
            meilleurScore = candidateScore;
            amelioration = true;
            break;
          }
        }

        if (amelioration) {
          break;
        }
      }

      if (amelioration) {
        break;
      }
    }

    iteration += 1;
  }

  return meilleureRoute;
}

function testerReecritureBlocMicroSecteur_(route, startPosition, endPosition, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain, scoreReference) {
  var source = route.slice(startPosition, endPosition + 1);
  var permutations = listerPermutationsTableau_(source);
  var meilleureRoute = null;
  var meilleurScore = scoreReference;
  var candidateRoute = null;
  var candidateScore = null;
  var permutation = null;
  var i = 0;

  for (i = 0; i < permutations.length; i += 1) {
    permutation = permutations[i];

    if (sontTableauxEgaux_(permutation, source)) {
      continue;
    }

    candidateRoute = route.slice();
    candidateRoute.splice.apply(candidateRoute, [startPosition, source.length].concat(permutation));
    candidateScore = evaluerParcours_(candidateRoute, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain);

    if (estMeilleurScore_(candidateScore, meilleurScore)) {
      meilleureRoute = candidateRoute;
      meilleurScore = candidateScore;
    }
  }

  if (!meilleureRoute) {
    return null;
  }

  return {
    route: meilleureRoute,
    score: meilleurScore
  };
}

function listerBlocsMicroSecteursContigus_(route, points, contexteUrbain) {
  var blocs = [];
  var startPosition = 0;
  var endPosition = 0;

  if (!route || route.length < 2) {
    return blocs;
  }

  startPosition = 0;
  endPosition = 0;

  while (startPosition < route.length) {
    endPosition = startPosition;

    while (endPosition + 1 < route.length && appartiennentAuMemeMicroSecteur_(route[endPosition], route[endPosition + 1], points, contexteUrbain)) {
      endPosition += 1;
    }

    if (endPosition > startPosition) {
      blocs.push({
        start: startPosition,
        end: endPosition
      });
    }

    startPosition = endPosition + 1;
  }

  return blocs;
}

function appartiennentAuMemeMicroSecteur_(indexA, indexB, points, contexteUrbain) {
  var bassinA = null;
  var bassinB = null;
  var rueA = '';
  var rueB = '';

  if (indexA === undefined || indexB === undefined || !points || !contexteUrbain) {
    return false;
  }

  bassinA = contexteUrbain.bassinParPoint ? contexteUrbain.bassinParPoint[construireClePoint_(points[indexA])] : null;
  bassinB = contexteUrbain.bassinParPoint ? contexteUrbain.bassinParPoint[construireClePoint_(points[indexB])] : null;

  if (bassinA !== undefined && bassinA !== null && bassinA === bassinB) {
    return true;
  }

  rueA = canoniserNomRue_(points[indexA] && points[indexA].adresse);
  rueB = canoniserNomRue_(points[indexB] && points[indexB].adresse);

  if (rueA && rueA === rueB && calculerDistanceMetres_(points[indexA], points[indexB]) <= 180) {
    return true;
  }

  return false;
}

function evaluerParcours_(route, durations, distances, startIndex, endIndex, pairesLocales, opportunitesInsertion, points, contexteUrbain) {
  var duration = 0;
  var distance = 0;
  var i = 0;

  if (route.length === 0) {
    return {
      duration: safeCost_(durations[startIndex][endIndex]),
      distance: safeCost_(distances[startIndex][endIndex]),
      opportunityPenalty: 0,
      localPenalty: 0,
      backtrackingPenalty: 0,
      sectorReentryPenalty: 0,
      appendixMidRoutePenalty: 0,
      proximityDetourPenalty: 0,
      twoStepLookaheadPenalty: 0,
      deferredPassagePenalty: 0
    };
  }

  duration += safeCost_(durations[startIndex][route[0]]);
  distance += safeCost_(distances[startIndex][route[0]]);

  for (i = 0; i < route.length - 1; i += 1) {
    duration += safeCost_(durations[route[i]][route[i + 1]]);
    distance += safeCost_(distances[route[i]][route[i + 1]]);
  }

  duration += safeCost_(durations[route[route.length - 1]][endIndex]);
  distance += safeCost_(distances[route[route.length - 1]][endIndex]);

  return {
    duration: duration,
    distance: distance,
    opportunityPenalty: calculerPenaliteOpportunite_(route, durations, startIndex, endIndex, opportunitesInsertion),
    localPenalty: calculerPenaliteLocale_(route, pairesLocales),
    backtrackingPenalty: points ? calculerPenaliteRetourArriere_(route, startIndex, endIndex, points) : 0,
    sectorReentryPenalty: (points && contexteUrbain) ? calculerPenaliteReentreeBassins_(route, points, contexteUrbain) : 0,
    appendixMidRoutePenalty: (points && contexteUrbain) ? calculerPenaliteAppendiceMilieu_(route, points, contexteUrbain) : 0,
    proximityDetourPenalty: (points && contexteUrbain) ? calculerPenaliteDetourProximite_(route, contexteUrbain) : 0,
    twoStepLookaheadPenalty: points ? calculerPenaliteLookaheadDeuxPoints_(route, durations, distances, startIndex, endIndex, points) : 0,
    deferredPassagePenalty: points ? calculerPenalitePassageDiffere_(route, durations, startIndex, endIndex, opportunitesInsertion) : 0
  };
}

function estMeilleurScore_(candidat, reference) {
  var toleranceDuree = 10;
  var toleranceDistance = 100;
  var toleranceOpportunite = 20;
  var toleranceLocale = 0.1;
  var toleranceBacktracking = 0.5;
  var toleranceReentreeBassin = 0.35;
  var toleranceAppendice = 0.35;
  var tolerancePassageDiffere = 0.5;
  var poidsDetourProximite = 12;
  var poidsLookahead = 10;
  var poidsBacktracking = 8;
  var dureePondereeCandidat = 0;
  var dureePondereeReference = 0;

  if (!reference) {
    return true;
  }

  dureePondereeCandidat = candidat.duration + (candidat.proximityDetourPenalty || 0) * poidsDetourProximite + (candidat.twoStepLookaheadPenalty || 0) * poidsLookahead + (candidat.backtrackingPenalty || 0) * poidsBacktracking;
  dureePondereeReference = reference.duration + (reference.proximityDetourPenalty || 0) * poidsDetourProximite + (reference.twoStepLookaheadPenalty || 0) * poidsLookahead + (reference.backtrackingPenalty || 0) * poidsBacktracking;

  if (dureePondereeCandidat + toleranceDuree < dureePondereeReference) {
    return true;
  }

  if (dureePondereeReference + toleranceDuree < dureePondereeCandidat) {
    return false;
  }

  if (candidat.sectorReentryPenalty !== undefined && reference.sectorReentryPenalty !== undefined) {
    if (candidat.sectorReentryPenalty + toleranceReentreeBassin < reference.sectorReentryPenalty) {
      return true;
    }

    if (reference.sectorReentryPenalty + toleranceReentreeBassin < candidat.sectorReentryPenalty) {
      return false;
    }
  }

  if (candidat.appendixMidRoutePenalty !== undefined && reference.appendixMidRoutePenalty !== undefined) {
    if (candidat.appendixMidRoutePenalty + toleranceAppendice < reference.appendixMidRoutePenalty) {
      return true;
    }

    if (reference.appendixMidRoutePenalty + toleranceAppendice < candidat.appendixMidRoutePenalty) {
      return false;
    }
  }

  if (candidat.deferredPassagePenalty !== undefined && reference.deferredPassagePenalty !== undefined) {
    if (candidat.deferredPassagePenalty + tolerancePassageDiffere < reference.deferredPassagePenalty) {
      return true;
    }

    if (reference.deferredPassagePenalty + tolerancePassageDiffere < candidat.deferredPassagePenalty) {
      return false;
    }
  }

  if (candidat.distance + toleranceDistance < reference.distance) {
    return true;
  }

  if (reference.distance + toleranceDistance < candidat.distance) {
    return false;
  }

  if (candidat.opportunityPenalty + toleranceOpportunite < reference.opportunityPenalty) {
    return true;
  }

  if (reference.opportunityPenalty + toleranceOpportunite < candidat.opportunityPenalty) {
    return false;
  }

  if (candidat.localPenalty + toleranceLocale < reference.localPenalty) {
    return true;
  }

  if (reference.localPenalty + toleranceLocale < candidat.localPenalty) {
    return false;
  }

  if (dureePondereeCandidat !== dureePondereeReference) {
    return dureePondereeCandidat < dureePondereeReference;
  }

  if (candidat.deferredPassagePenalty !== undefined && reference.deferredPassagePenalty !== undefined && candidat.deferredPassagePenalty !== reference.deferredPassagePenalty) {
    return candidat.deferredPassagePenalty < reference.deferredPassagePenalty;
  }

  if (candidat.distance !== reference.distance) {
    return candidat.distance < reference.distance;
  }

  if (candidat.opportunityPenalty !== reference.opportunityPenalty) {
    return candidat.opportunityPenalty < reference.opportunityPenalty;
  }

  if (candidat.localPenalty !== reference.localPenalty) {
    return candidat.localPenalty < reference.localPenalty;
  }

  if (candidat.sectorReentryPenalty !== undefined && reference.sectorReentryPenalty !== undefined && candidat.sectorReentryPenalty !== reference.sectorReentryPenalty) {
    return candidat.sectorReentryPenalty < reference.sectorReentryPenalty;
  }

  if (candidat.appendixMidRoutePenalty !== undefined && reference.appendixMidRoutePenalty !== undefined && candidat.appendixMidRoutePenalty !== reference.appendixMidRoutePenalty) {
    return candidat.appendixMidRoutePenalty < reference.appendixMidRoutePenalty;
  }

  if (candidat.backtrackingPenalty !== undefined && reference.backtrackingPenalty !== undefined && candidat.backtrackingPenalty !== reference.backtrackingPenalty) {
    return candidat.backtrackingPenalty < reference.backtrackingPenalty;
  }

  return false;
}

function construirePairesLocales_(points, durations, distances, startIndex, endIndex) {
  var paires = [];
  var i = 0;
  var j = 0;
  var distanceGeo = 0;
  var dureeDirecte = 0;
  var distanceDirecte = 0;
  var poids = 0;

  for (i = startIndex + 1; i < endIndex; i += 1) {
    for (j = i + 1; j < endIndex; j += 1) {
      distanceGeo = calculerDistanceMetres_(points[i], points[j]);
      dureeDirecte = Math.min(safeCost_(durations[i][j]), safeCost_(durations[j][i]));
      distanceDirecte = Math.min(safeCost_(distances[i][j]), safeCost_(distances[j][i]));

      if (distanceGeo <= 120 && dureeDirecte <= 120 && distanceDirecte <= 350) {
        poids = (1 + ((120 - distanceGeo) / 120)) + (1 + ((120 - dureeDirecte) / 120));
        paires.push({
          a: i,
          b: j,
          weight: poids
        });
      }
    }
  }

  return paires;
}

function construireArcsDetourProximite_(points, durations, distances, startIndex, endIndex) {
  var arcs = {};
  var rues = {};
  var distanceGeo = 0;
  var distanceArc = 0;
  var dureeArc = 0;
  var dureeInverse = 0;
  var memeRue = false;
  var penalite = 0;
  var penaliteAsymetrie = 0;
  var detourEtendu = false;
  var distanceArcInverse = 0;
  var ratioDetourMax = 0;
  var i = 0;
  var j = 0;

  if (!points || !durations || !distances) {
    return arcs;
  }

  for (i = startIndex + 1; i < endIndex; i += 1) {
    rues[i] = canoniserNomRue_(points[i] && points[i].adresse);
  }

  for (i = startIndex + 1; i < endIndex; i += 1) {
    for (j = i + 1; j < endIndex; j += 1) {
      distanceGeo = calculerDistanceMetres_(points[i], points[j]);
      memeRue = !!rues[i] && rues[i] === rues[j];

      if (distanceGeo > (memeRue ? 380 : 250)) {
        continue;
      }

      detourEtendu = !memeRue && distanceGeo > 90;
      distanceArc = safeCost_(distances[i][j]);
      dureeArc = safeCost_(durations[i][j]);
      dureeInverse = safeCost_(durations[j][i]);

      if (detourEtendu) {
        distanceArcInverse = safeCost_(distances[j][i]);
        ratioDetourMax = Math.max(
          isFinite(distanceArc) ? distanceArc / Math.max(distanceGeo, 1) : 0,
          isFinite(distanceArcInverse) ? distanceArcInverse / Math.max(distanceGeo, 1) : 0
        );
        if (ratioDetourMax < 4) {
          continue;
        }
      }

      penalite = calculerPoidsDetourProximiteArc_(distanceGeo, distanceArc, dureeArc, memeRue);
      penaliteAsymetrie = memeRue ? calculerPoidsAsymetrieArc_(dureeArc, dureeInverse) : 0;
      if (Math.max(penalite, penaliteAsymetrie) > 0) {
        arcs[construireCleArc_(i, j)] = Math.max(penalite, penaliteAsymetrie);
      }

      distanceArc = safeCost_(distances[j][i]);
      dureeArc = safeCost_(durations[j][i]);
      dureeInverse = safeCost_(durations[i][j]);

      penalite = calculerPoidsDetourProximiteArc_(distanceGeo, distanceArc, dureeArc, memeRue);
      penaliteAsymetrie = memeRue ? calculerPoidsAsymetrieArc_(dureeArc, dureeInverse) : 0;
      if (Math.max(penalite, penaliteAsymetrie) > 0) {
        arcs[construireCleArc_(j, i)] = Math.max(penalite, penaliteAsymetrie);
      }
    }
  }

  return arcs;
}

function calculerPoidsAsymetrieArc_(dureeArc, dureeInverse) {
  var ratio = 0;
  var poids = 0;

  if (!isFinite(dureeArc) || !isFinite(dureeInverse) || dureeInverse <= 0 || dureeArc <= 0) {
    return 0;
  }

  if (dureeArc <= dureeInverse) {
    return 0;
  }

  ratio = dureeArc / dureeInverse;

  if (ratio < 2.2) {
    return 0;
  }

  poids = Math.min(8, (ratio - 2.2) * 1.8 + 1.5);

  if (dureeArc >= 90) {
    poids += Math.min(3, (dureeArc - 90) / 55);
  }

  return poids;
}

function calculerPoidsDetourProximiteArc_(distanceGeo, distanceArc, dureeArc, memeRue) {
  var seuilRatio = memeRue ? 4 : 5;
  var seuilDistance = memeRue ? 260 : 320;
  var seuilDuree = memeRue ? 55 : 70;
  var ratio = 0;
  var poids = 0;

  if (!isFinite(distanceGeo) || distanceGeo <= 0 || !isFinite(distanceArc) || !isFinite(dureeArc)) {
    return 0;
  }

  ratio = distanceArc / Math.max(distanceGeo, 1);

  if (distanceArc <= seuilDistance && ratio <= seuilRatio && dureeArc <= seuilDuree) {
    return 0;
  }

  if (ratio > seuilRatio) {
    poids += Math.min(6, (ratio - seuilRatio) / 1.2);
  }

  if (distanceArc > seuilDistance) {
    poids += Math.min(4, (distanceArc - seuilDistance) / 180);
  }

  if (dureeArc > seuilDuree) {
    poids += Math.min(3, (dureeArc - seuilDuree) / 45);
  }

  if (memeRue) {
    poids *= 1.25;
  }

  return poids >= (memeRue ? 1.25 : 1.2) ? poids : 0;
}

function construireOpportunitesInsertion_(points, durations, startIndex, endIndex) {
  var opportunites = {};
  var maxOpportunites = 6;
  var node = 0;
  var prev = 0;
  var next = 0;
  var liste = null;
  var listeSecours = null;
  var deltaDuration = 0;
  var distancePrev = 0;
  var distanceNext = 0;
  var analyseSegment = null;
  var continuiteRue = 0;
  var poidsPassage = 0;

  for (node = startIndex + 1; node < endIndex; node += 1) {
    liste = [];
    listeSecours = [];

    for (prev = startIndex; prev < endIndex; prev += 1) {
      if (prev === node) {
        continue;
      }

      for (next = startIndex + 1; next <= endIndex; next += 1) {
        if (next === node || next === prev) {
          continue;
        }

        deltaDuration = calculerDeltaInsertion_(prev, node, next, durations);
        if (!isFinite(deltaDuration)) {
          continue;
        }

        distancePrev = calculerDistanceMetres_(points[prev], points[node]);
        distanceNext = calculerDistanceMetres_(points[node], points[next]);
        analyseSegment = analyserProjectionPointSegmentMetres_(points[node], points[prev], points[next]);
        continuiteRue = evaluerContinuiteRueSegment_(points[node], points[prev], points[next]);
        poidsPassage = calculerPoidsPassageUtile_(analyseSegment, deltaDuration, continuiteRue);

        listeSecours.push({
          prev: prev,
          next: next,
          deltaDuration: deltaDuration,
          passageWeight: poidsPassage
        });

        if (deltaDuration <= 180 || Math.min(distancePrev, distanceNext) <= 220 || (distancePrev + distanceNext) <= 450) {
          liste.push({
            prev: prev,
            next: next,
            deltaDuration: deltaDuration,
            passageWeight: poidsPassage
          });
        }
      }
    }

    if (liste.length === 0) {
      liste = listeSecours;
    }

    liste.sort(function(a, b) {
      return (a.deltaDuration - (a.passageWeight || 0) * 22) - (b.deltaDuration - (b.passageWeight || 0) * 22);
    });

    opportunites[node] = liste.slice(0, maxOpportunites);
  }

  return opportunites;
}

function calculerPenaliteOpportunite_(route, durations, startIndex, endIndex, opportunitesInsertion) {
  var parcours = [startIndex].concat(route).concat([endIndex]);
  var adjacences = {};
  var penalite = 0;
  var position = 0;
  var node = 0;
  var actualDelta = 0;
  var meilleurDelta = Infinity;
  var opportunites = null;
  var i = 0;
  var diff = 0;

  if (!opportunitesInsertion) {
    return 0;
  }

  for (i = 0; i < parcours.length - 1; i += 1) {
    adjacences[construireCleArc_(parcours[i], parcours[i + 1])] = true;
  }

  for (position = 1; position < parcours.length - 1; position += 1) {
    node = parcours[position];
    opportunites = opportunitesInsertion[node] || [];
    actualDelta = calculerDeltaInsertion_(parcours[position - 1], node, parcours[position + 1], durations);
    meilleurDelta = Infinity;

    for (i = 0; i < opportunites.length; i += 1) {
      if (adjacences[construireCleArc_(opportunites[i].prev, opportunites[i].next)]) {
        if (opportunites[i].deltaDuration < meilleurDelta) {
          meilleurDelta = opportunites[i].deltaDuration;
        }
      }
    }

    if (meilleurDelta < Infinity && actualDelta > meilleurDelta + 5) {
      diff = actualDelta - meilleurDelta;
      penalite += Math.min(diff, 180);
    }
  }

  return penalite;
}

function calculerPoidsPassageUtile_(analyseSegment, deltaDuration, continuiteRue) {
  var deltaMax = continuiteRue > 0 ? 160 : 95;
  var poids = 0;
  var projectionInterieure = false;
  var distanceGeo = analyseSegment ? analyseSegment.distance : Number.POSITIVE_INFINITY;
  var distanceMax = continuiteRue > 0 ? 90 : 45;

  if (!isFinite(deltaDuration) || deltaDuration > deltaMax) {
    return 0;
  }

  if (continuiteRue > 0) {
    poids = 1;
    poids += Math.max(0, (deltaMax - deltaDuration) / 80) * 1.1;
    poids += continuiteRue * 1.0;

    if (analyseSegment && isFinite(distanceGeo) && distanceGeo <= distanceMax) {
      projectionInterieure = analyseSegment.projection >= 0.08 && analyseSegment.projection <= 0.92;
      if (projectionInterieure) {
        poids += Math.max(0, (distanceMax - distanceGeo) / 45) * 0.8;
        if (distanceGeo <= 22) {
          poids += 0.9;
        }
      }
    }

    return poids;
  }

  if (!analyseSegment || !isFinite(distanceGeo) || distanceGeo > distanceMax) {
    return 0;
  }

  projectionInterieure = analyseSegment.projection >= 0.12 && analyseSegment.projection <= 0.88;
  if (!projectionInterieure) {
    return 0;
  }

  poids = 1;
  poids += Math.max(0, (distanceMax - distanceGeo) / 22);
  poids += Math.max(0, (deltaMax - deltaDuration) / 75) * 0.7;

  if (distanceGeo <= 18) {
    poids += 0.8;
  }

  return poids;
}

function calculerPenalitePassageDiffere_(route, durations, startIndex, endIndex, opportunitesInsertion) {
  var parcours = [startIndex].concat(route).concat([endIndex]);
  var positionsArcs = {};
  var penalite = 0;
  var position = 0;
  var node = 0;
  var actualDelta = 0;
  var opportunites = null;
  var meilleureOpportunity = null;
  var positionArc = -1;
  var retard = 0;
  var diff = 0;
  var i = 0;
  var cleArc = '';

  if (!opportunitesInsertion) {
    return 0;
  }

  for (i = 0; i < parcours.length - 1; i += 1) {
    cleArc = construireCleArc_(parcours[i], parcours[i + 1]);
    if (positionsArcs[cleArc] === undefined) {
      positionsArcs[cleArc] = i;
    }
  }

  for (position = 1; position < parcours.length - 1; position += 1) {
    node = parcours[position];
    actualDelta = calculerDeltaInsertion_(parcours[position - 1], node, parcours[position + 1], durations);
    opportunites = opportunitesInsertion[node] || [];
    meilleureOpportunity = null;
    positionArc = -1;

    for (i = 0; i < opportunites.length; i += 1) {
      if (!opportunites[i].passageWeight) {
        continue;
      }

      cleArc = construireCleArc_(opportunites[i].prev, opportunites[i].next);
      if (positionsArcs[cleArc] === undefined || positionsArcs[cleArc] >= position - 1) {
        continue;
      }

      if (opportunites[i].deltaDuration > actualDelta + 45) {
        continue;
      }

      if (!meilleureOpportunity || opportunites[i].passageWeight > meilleureOpportunity.passageWeight + 0.15 || (Math.abs(opportunites[i].passageWeight - meilleureOpportunity.passageWeight) <= 0.15 && opportunites[i].deltaDuration < meilleureOpportunity.deltaDuration)) {
        meilleureOpportunity = opportunites[i];
        positionArc = positionsArcs[cleArc];
      }
    }

    if (!meilleureOpportunity) {
      continue;
    }

    retard = position - (positionArc + 1);
    if (retard <= 1) {
      continue;
    }

    diff = Math.max(0, actualDelta - meilleureOpportunity.deltaDuration);
    penalite += meilleureOpportunity.passageWeight * (0.8 + Math.min(3, (retard - 1) * 0.65)) + Math.min(2.5, diff / 35);
  }

  return penalite;
}

function calculerPenaliteLocale_(route, pairesLocales) {
  var positions = {};
  var penalite = 0;
  var i = 0;
  var ecart = 0;

  if (!pairesLocales || pairesLocales.length === 0) {
    return 0;
  }

  for (i = 0; i < route.length; i += 1) {
    positions[route[i]] = i;
  }

  for (i = 0; i < pairesLocales.length; i += 1) {
    ecart = Math.abs(positions[pairesLocales[i].a] - positions[pairesLocales[i].b]);

    if (ecart > 1) {
      penalite += (ecart - 1) * pairesLocales[i].weight;
    }
  }

  return penalite;
}

function calculerPenaliteRetourArriere_(route, startIndex, endIndex, points) {
  var penalite = 0;
  var parcours = [startIndex].concat(route).concat([endIndex]);
  var i = 0;
  var pointA = null;
  var pointB = null;
  var pointC = null;
  var vecteurAB = null;
  var vecteurBC = null;
  var produitScalaire = 0;
  var normeAB = 0;
  var normeBC = 0;
  var cosAngle = 0;
  var angle = 0;

  if (parcours.length < 3) {
    return 0;
  }

  for (i = 0; i < parcours.length - 2; i += 1) {
    pointA = points[parcours[i]];
    pointB = points[parcours[i + 1]];
    pointC = points[parcours[i + 2]];

    if (!pointA || !pointB || !pointC) {
      continue;
    }

    vecteurAB = {
      lat: pointB.latitude - pointA.latitude,
      lon: pointB.longitude - pointA.longitude
    };

    vecteurBC = {
      lat: pointC.latitude - pointB.latitude,
      lon: pointC.longitude - pointB.longitude
    };

    produitScalaire = vecteurAB.lat * vecteurBC.lat + vecteurAB.lon * vecteurBC.lon;
    normeAB = Math.sqrt(vecteurAB.lat * vecteurAB.lat + vecteurAB.lon * vecteurAB.lon);
    normeBC = Math.sqrt(vecteurBC.lat * vecteurBC.lat + vecteurBC.lon * vecteurBC.lon);

    if (normeAB === 0 || normeBC === 0) {
      continue;
    }

    cosAngle = produitScalaire / (normeAB * normeBC);
    cosAngle = Math.max(-1, Math.min(1, cosAngle));
    angle = Math.acos(cosAngle) * (180 / Math.PI);

    if (angle > 100) {
      penalite += (angle - 100) / 40;
    }
  }

  return penalite;
}

function calculerPenaliteReentreeBassins_(route, points, contexteUrbain) {
  var sequence = [];
  var visites = {};
  var bassin = null;
  var precedent = null;
  var penalite = 0;
  var i = 0;

  if (!contexteUrbain || !contexteUrbain.bassinParPoint) {
    return 0;
  }

  for (i = 0; i < route.length; i += 1) {
    bassin = contexteUrbain.bassinParPoint[construireClePoint_(points[route[i]])];

    if (bassin === undefined || bassin === null) {
      continue;
    }

    if (sequence.length === 0 || sequence[sequence.length - 1] !== bassin) {
      sequence.push(bassin);
    }
  }

  for (i = 0; i < sequence.length; i += 1) {
    bassin = sequence[i];

    if (visites[bassin] && precedent !== bassin) {
      penalite += 1;
    }

    visites[bassin] = true;
    precedent = bassin;
  }

  return penalite;
}

function calculerPenaliteAppendiceMilieu_(route, points, contexteUrbain) {
  var penalite = 0;
  var bassinCourant = null;
  var bassinPrecedent = null;
  var bassinSuivant = null;
  var i = 0;

  if (!contexteUrbain || !contexteUrbain.appendixIndexes) {
    return 0;
  }

  for (i = 0; i < route.length; i += 1) {
    if (!contexteUrbain.appendixIndexes[route[i]]) {
      continue;
    }

    if (i === 0 || i === route.length - 1) {
      continue;
    }

    bassinCourant = contexteUrbain.bassinParPoint[construireClePoint_(points[route[i]])];
    bassinPrecedent = i > 0 ? contexteUrbain.bassinParPoint[construireClePoint_(points[route[i - 1]])] : null;
    bassinSuivant = i < route.length - 1 ? contexteUrbain.bassinParPoint[construireClePoint_(points[route[i + 1]])] : null;

    if (bassinPrecedent === bassinCourant && bassinSuivant === bassinCourant) {
      penalite += 0.8;
    } else if (bassinPrecedent !== bassinCourant && bassinSuivant !== bassinCourant) {
      penalite += 1.0;
    }
  }

  return penalite;
}

function twoOptSwap_(route, i, k) {
  return route.slice(0, i)
    .concat(route.slice(i, k + 1).reverse())
    .concat(route.slice(k + 1));
}

function calculerDeltaInsertion_(prev, node, next, durations) {
  return safeCost_(durations[prev][node]) + safeCost_(durations[node][next]) - safeCost_(durations[prev][next]);
}

function construireCleArc_(fromIndex, toIndex) {
  return String(fromIndex) + '>' + String(toIndex);
}

function ajouterUnique_(array, value) {
  if (value === undefined || value === null) {
    return;
  }

  if (array.indexOf(value) === -1) {
    array.push(value);
  }
}

function ecrireTourneesPlanifiees_(spreadsheet, tournees, feuillesAStandardiser, feuillesAttendues) {
  var feuillesUtilisees = {};
  var i = 0;

  for (i = 0; i < tournees.length; i += 1) {
    ecrireFeuilleTournee_(spreadsheet, tournees[i].points, tournees[i].ordreInterne, tournees[i].matrix, tournees[i].sheetName);
    feuillesUtilisees[tournees[i].sheetName] = true;
  }

  (feuillesAttendues || []).forEach(function(sheetName) {
    if (!feuillesUtilisees[sheetName]) {
      standardiserFeuilleTournee_(obtenirOuCreerFeuille_(spreadsheet, sheetName));
    }
  });

  (feuillesAStandardiser || []).forEach(function(sheetName) {
    standardiserFeuilleTournee_(obtenirOuCreerFeuille_(spreadsheet, sheetName));
  });
}

function effacerFeuillesTournees_(spreadsheet, sheetNames) {
  (sheetNames || ['Tournee_J1', 'Tournee_J2']).forEach(function(sheetName) {
    var sheet = obtenirOuCreerFeuille_(spreadsheet, sheetName);

    if (sheet.getLastRow() > 1) {
      sheet.getRange(2, 1, sheet.getLastRow() - 1, sheet.getMaxColumns()).clearContent();
    }

    standardiserFeuilleTournee_(sheet);
  });
}

function ecrireFeuilleTournee_(spreadsheet, points, ordreInterne, matrix, sheetName) {
  var feuilleCible = sheetName || 'Tournee_J1';
  var sheet = spreadsheet.getSheetByName(feuilleCible);
  if (!sheet) {
    sheet = spreadsheet.insertSheet(feuilleCible);
  }

  sheet.clearContents();

  var headers = [
    'Ordre',
    'ID',
    'Adresse',
    'Latitude',
    'Longitude',
    'Distance_depuis_precedent_km',
    'Duree_depuis_precedent_min',
    'Distance_cumulee_km',
    'Duree_cumulee_min'
  ];

  var orderedPointIndexes = [0].concat(ordreInterne).concat([points.length - 1]);
  var rows = [headers];
  var cumulativeDistance = 0;
  var cumulativeDuration = 0;

  for (var i = 0; i < orderedPointIndexes.length; i += 1) {
    var pointIndex = orderedPointIndexes[i];
    var point = points[pointIndex];
    var previousIndex = i === 0 ? null : orderedPointIndexes[i - 1];
    var legDistance = previousIndex === null ? 0 : safeCost_(matrix.distances[previousIndex][pointIndex]);
    var legDuration = previousIndex === null ? 0 : safeCost_(matrix.durations[previousIndex][pointIndex]);

    cumulativeDistance += legDistance;
    cumulativeDuration += legDuration;

    var ordre = '';
    if (i > 0 && i < orderedPointIndexes.length - 1) {
      ordre = i;
    }

    rows.push([
      ordre,
      point.id,
      point.adresse,
      point.latitude,
      point.longitude,
      roundTo_(legDistance / 1000, 2),
      roundTo_(legDuration / 60, 1),
      roundTo_(cumulativeDistance / 1000, 2),
      roundTo_(cumulativeDuration / 60, 1)
    ]);
  }

  sheet.getRange(1, 1, rows.length, headers.length).setValues(rows);
  standardiserFeuilleTournee_(sheet);
}

function getHeaderMap_(headersRow) {
  var map = {};
  for (var i = 0; i < headersRow.length; i += 1) {
    var normalized = normalizeHeader_(headersRow[i]);
    if (normalized) {
      map[normalized] = i;
    }
  }
  return map;
}

function normalizeHeader_(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function toDisplayHeader_(header) {
  return header
    .split('_')
    .map(function(part) {
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join('_');
}

function toBoolean_(value) {
  if (typeof value === 'boolean') {
    return value;
  }

  var normalized = String(value || '').trim().toLowerCase();
  return normalized === 'true' || normalized === 'oui' || normalized === '1' || normalized === 'x';
}

function toNumber_(value, fieldName, rowNumber) {
  var normalized = String(value).replace(',', '.').trim();
  var number = Number(normalized);

  if (!isFinite(number)) {
    throw new Error('Valeur invalide pour ' + fieldName + ' à la ligne ' + rowNumber + '.');
  }

  return number;
}

function safeCost_(value) {
  if (value === null || value === undefined || value === '') {
    return Number.POSITIVE_INFINITY;
  }

  var number = Number(value);
  if (!isFinite(number)) {
    return Number.POSITIVE_INFINITY;
  }

  return number;
}

function roundTo_(value, decimals) {
  var factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}

function calculerDistanceMetres_(pointA, pointB) {
  var earthRadius = 6371000;
  var lat1 = degreesToRadians_(pointA.latitude);
  var lat2 = degreesToRadians_(pointB.latitude);
  var deltaLat = degreesToRadians_(pointB.latitude - pointA.latitude);
  var deltaLon = degreesToRadians_(pointB.longitude - pointA.longitude);
  var sinLat = Math.sin(deltaLat / 2);
  var sinLon = Math.sin(deltaLon / 2);
  var a = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLon * sinLon;
  var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return earthRadius * c;
}

function degreesToRadians_(degrees) {
  return degrees * Math.PI / 180;
}

function initialiserClasseur_(spreadsheet) {
  var horodateursSheet = obtenirOuCreerFeuille_(spreadsheet, 'Horodateurs');
  var parametresSheet = obtenirOuCreerFeuille_(spreadsheet, 'Parametres');
  var tourneeJ1Sheet = obtenirOuCreerFeuille_(spreadsheet, 'Tournee_J1');
  var tourneeJ2Sheet = obtenirOuCreerFeuille_(spreadsheet, 'Tournee_J2');
  var aideSheet = obtenirOuCreerFeuille_(spreadsheet, 'Aide');

  supprimerFeuillesSiExistantes_(spreadsheet, [
    'Tournee',
    'Comparatif_V2_J1',
    'Comparatif_V2_J2',
    'Comparatif_V3_J1',
    'Comparatif_V3_J2'
  ]);

  standardiserFeuilleHorodateurs_(horodateursSheet);
  standardiserFeuilleParametres_(parametresSheet);
  standardiserFeuilleTournee_(tourneeJ1Sheet);
  standardiserFeuilleTournee_(tourneeJ2Sheet);
  standardiserFeuilleAide_(aideSheet);
}

function obtenirOuCreerFeuille_(spreadsheet, sheetName) {
  var sheet = spreadsheet.getSheetByName(sheetName);

  if (!sheet) {
    sheet = spreadsheet.insertSheet(sheetName);
  }

  return sheet;
}

function supprimerFeuillesSiExistantes_(spreadsheet, sheetNames) {
  (sheetNames || []).forEach(function(sheetName) {
    var sheet = spreadsheet.getSheetByName(sheetName);

    if (sheet) {
      spreadsheet.deleteSheet(sheet);
    }
  });
}

function supprimerColonneOptionnelleOrdre_(sheet) {
  var headerValues = sheet.getRange(1, 1, 1, sheet.getMaxColumns()).getValues()[0];
  var headers = getHeaderMap_(headerValues);

  if (headers.ordre !== undefined) {
    sheet.deleteColumn(headers.ordre + 1);
  }
}

function standardiserFeuilleHorodateurs_(sheet) {
  var headers = ['ID', 'Adresse', 'Latitude', 'Longitude', 'Selection'];
  var dataRowCount = Math.max(sheet.getMaxRows() - 1, 1);
  var dataRange = null;
  var selectionRange = null;

  supprimerColonneOptionnelleOrdre_(sheet);
  assurerDimensionsMinimales_(sheet, 200, headers.length);
  sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
  appliquerStyleEntete_(sheet.getRange(1, 1, 1, headers.length));
  sheet.setFrozenRows(1);
  sheet.setColumnWidth(1, 120);
  sheet.setColumnWidth(2, 320);
  sheet.setColumnWidth(3, 110);
  sheet.setColumnWidth(4, 110);
  sheet.setColumnWidth(5, 110);

  dataRange = sheet.getRange(2, 1, dataRowCount, headers.length);
  selectionRange = sheet.getRange(2, 5, dataRowCount, 1);

  dataRange.setVerticalAlignment('middle');
  sheet.getRange(2, 1, dataRowCount, 2).setWrap(true);
  sheet.getRange(2, 3, dataRowCount, 2).setNumberFormat('0.000000');
  sheet.getRange(2, 5, dataRowCount, 1).setHorizontalAlignment('center');

  sheet.getRange(2, 3, dataRowCount, 1)
    .setDataValidation(
      SpreadsheetApp.newDataValidation()
        .requireNumberBetween(-90, 90)
        .setAllowInvalid(false)
        .setHelpText('Latitude comprise entre -90 et 90.')
        .build()
    );

  sheet.getRange(2, 4, dataRowCount, 1)
    .setDataValidation(
      SpreadsheetApp.newDataValidation()
        .requireNumberBetween(-180, 180)
        .setAllowInvalid(false)
        .setHelpText('Longitude comprise entre -180 et 180.')
        .build()
    );

  selectionRange.clearDataValidations();
  selectionRange.insertCheckboxes();
  creerOuRemplacerFiltre_(sheet, headers.length);
  appliquerBanding_(sheet, headers.length);
  appliquerMiseEnFormeConditionnelleHorodateurs_(sheet, headers.length);
}

function standardiserFeuilleParametres_(sheet) {
  var headers = [
    'Depart_ID',
    'Depart_Adresse',
    'Depart_Latitude',
    'Depart_Longitude',
    'Arrivee_ID',
    'Arrivee_Adresse',
    'Arrivee_Latitude',
    'Arrivee_Longitude',
    'Profil_ORS'
  ];
  var values = null;

  assurerDimensionsMinimales_(sheet, 20, headers.length);
  sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
  appliquerStyleEntete_(sheet.getRange(1, 1, 1, headers.length));
  sheet.setFrozenRows(1);
  sheet.setColumnWidth(1, 120);
  sheet.setColumnWidth(2, 260);
  sheet.setColumnWidth(3, 120);
  sheet.setColumnWidth(4, 120);
  sheet.setColumnWidth(5, 120);
  sheet.setColumnWidth(6, 260);
  sheet.setColumnWidth(7, 120);
  sheet.setColumnWidth(8, 120);
  sheet.setColumnWidth(9, 130);

  values = sheet.getRange(2, 1, 1, headers.length).getValues()[0];
  values[0] = String(values[0] || '').trim() || 'DEPART';
  values[4] = String(values[4] || '').trim() || 'ARRIVEE';
  values[8] = String(values[8] || '').trim() || 'driving-car';
  sheet.getRange(2, 1, 1, headers.length).setValues([values]);

  sheet.getRange(2, 1, sheet.getMaxRows() - 1, headers.length).setVerticalAlignment('middle');
  sheet.getRange(2, 1, sheet.getMaxRows() - 1, 2).setWrap(true);
  sheet.getRange(2, 5, sheet.getMaxRows() - 1, 2).setWrap(true);
  sheet.getRange(2, 3, sheet.getMaxRows() - 1, 2).setNumberFormat('0.000000');
  sheet.getRange(2, 7, sheet.getMaxRows() - 1, 2).setNumberFormat('0.000000');

  sheet.getRange(2, 3, sheet.getMaxRows() - 1, 1)
    .setDataValidation(
      SpreadsheetApp.newDataValidation()
        .requireNumberBetween(-90, 90)
        .setAllowInvalid(false)
        .setHelpText('Latitude de départ comprise entre -90 et 90.')
        .build()
    );

  sheet.getRange(2, 4, sheet.getMaxRows() - 1, 1)
    .setDataValidation(
      SpreadsheetApp.newDataValidation()
        .requireNumberBetween(-180, 180)
        .setAllowInvalid(false)
        .setHelpText('Longitude de départ comprise entre -180 et 180.')
        .build()
    );

  sheet.getRange(2, 7, sheet.getMaxRows() - 1, 1)
    .setDataValidation(
      SpreadsheetApp.newDataValidation()
        .requireNumberBetween(-90, 90)
        .setAllowInvalid(false)
        .setHelpText('Latitude d\'arrivée comprise entre -90 et 90.')
        .build()
    );

  sheet.getRange(2, 8, sheet.getMaxRows() - 1, 1)
    .setDataValidation(
      SpreadsheetApp.newDataValidation()
        .requireNumberBetween(-180, 180)
        .setAllowInvalid(false)
        .setHelpText('Longitude d\'arrivée comprise entre -180 et 180.')
        .build()
    );

  sheet.getRange(2, 9, sheet.getMaxRows() - 1, 1)
    .setDataValidation(
      SpreadsheetApp.newDataValidation()
        .requireValueInList(['driving-car', 'driving-hgv', 'cycling-regular', 'foot-walking'], true)
        .setAllowInvalid(false)
        .setHelpText('Profil ORS autorisé.')
        .build()
    );

  appliquerBanding_(sheet, headers.length);
}

function standardiserFeuilleTournee_(sheet) {
  var headers = [
    'Ordre',
    'ID',
    'Adresse',
    'Latitude',
    'Longitude',
    'Distance_depuis_precedent_km',
    'Duree_depuis_precedent_min',
    'Distance_cumulee_km',
    'Duree_cumulee_min'
  ];
  var rowCount = Math.max(sheet.getMaxRows() - 1, 1);

  assurerDimensionsMinimales_(sheet, 200, headers.length);
  sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
  appliquerStyleEntete_(sheet.getRange(1, 1, 1, headers.length));
  sheet.setFrozenRows(1);
  sheet.setColumnWidth(1, 90);
  sheet.setColumnWidth(2, 120);
  sheet.setColumnWidth(3, 320);
  sheet.setColumnWidth(4, 110);
  sheet.setColumnWidth(5, 110);
  sheet.setColumnWidth(6, 180);
  sheet.setColumnWidth(7, 170);
  sheet.setColumnWidth(8, 150);
  sheet.setColumnWidth(9, 150);

  sheet.getRange(2, 1, rowCount, headers.length).setVerticalAlignment('middle');
  sheet.getRange(2, 1, rowCount, 1).setHorizontalAlignment('center');
  sheet.getRange(2, 3, rowCount, 1).setWrap(true);
  sheet.getRange(2, 4, rowCount, 2).setNumberFormat('0.000000');
  sheet.getRange(2, 6, rowCount, 4).setNumberFormat('0.00');
  appliquerBanding_(sheet, headers.length);
}

function standardiserFeuilleAide_(sheet) {
  var rows = [
    ['Collecte des horodateurs - mode d\'emploi', ''],
    ['Etape', 'Instruction'],
    ['1', 'Dans la feuille Horodateurs, renseignez ID, Adresse, Latitude et Longitude pour chaque horodateur.'],
    ['2', 'Cochez la colonne Selection pour les horodateurs à collecter cette semaine. Les lignes sélectionnées apparaissent en vert.'],
    ['3', 'Dans la feuille Parametres, renseignez le départ, l\'arrivée et le profil ORS.'],
    ['4', 'Utilisez le menu Collecte > Configurer la clé ORS pour enregistrer la clé API.'],
    ['5', 'Utilisez le menu Collecte > Calculer la tournée pour générer les feuilles Tournee_J1 et Tournee_J2.'],
    ['6', 'Utilisez le menu Collecte > Réinitialiser la sélection pour décocher toutes les lignes si nécessaire.'],
    ['7', 'Imprimez la feuille Tournee_J1 ou Tournee_J2 pour le chauffeur selon la journée concernée.'],
    ['', ''],
    ['Rappels', ''],
    ['Résultat', 'La feuille Horodateurs reste une feuille de saisie. Le résultat de l\'optimisation est écrit dans Tournee_J1 et Tournee_J2.'],
    ['Couleurs', 'Une ligne sélectionnée est colorée en vert. Une ligne avec latitude ou longitude manquante est colorée en rouge.'],
    ['Coordonnées', 'Les feuilles utilisent Latitude puis Longitude, mais le script envoie automatiquement Longitude puis Latitude à ORS.'],
    ['Validation', 'Les latitudes, longitudes, cases à cocher et profils ORS sont contrôlés automatiquement.'],
    ['Conseil', 'Testez d\'abord avec 5 à 10 horodateurs avant un usage complet.']
  ];
  var rappelsRowIndex = 0;
  var i = 0;

  assurerDimensionsMinimales_(sheet, rows.length + 10, 2);
  sheet.getRange(1, 1, sheet.getMaxRows(), 2).breakApart();
  sheet.clearContents();
  sheet.clearFormats();
  sheet.getRange(1, 1, rows.length, 2).setValues(rows);
  sheet.setFrozenRows(2);
  sheet.setColumnWidth(1, 120);
  sheet.setColumnWidth(2, 700);
  sheet.getRange(1, 1, 1, 2)
    .merge()
    .setBackground('#1f4e78')
    .setFontColor('#ffffff')
    .setFontWeight('bold')
    .setHorizontalAlignment('center')
    .setVerticalAlignment('middle');
  appliquerStyleEntete_(sheet.getRange(2, 1, 1, 2));
  sheet.getRange(3, 1, rows.length - 2, 2).setVerticalAlignment('middle');
  sheet.getRange(3, 2, rows.length - 2, 1).setWrap(true);

  for (i = 0; i < rows.length; i += 1) {
    if (rows[i][0] === 'Rappels') {
      rappelsRowIndex = i + 1;
      break;
    }
  }

  if (rappelsRowIndex > 0) {
    sheet.getRange(rappelsRowIndex, 1, 1, 2).setFontWeight('bold').setBackground('#d9e2f3');
  }
}

function appliquerStyleEntete_(range) {
  range
    .setBackground('#1f4e78')
    .setFontColor('#ffffff')
    .setFontWeight('bold')
    .setHorizontalAlignment('center')
    .setVerticalAlignment('middle');
}

function appliquerBanding_(sheet, columnCount) {
  var bandings = sheet.getBandings();
  var range = sheet.getRange(1, 1, Math.max(sheet.getMaxRows(), 2), columnCount);
  var i = 0;

  for (i = 0; i < bandings.length; i += 1) {
    bandings[i].remove();
  }

  range.applyRowBanding(SpreadsheetApp.BandingTheme.LIGHT_GREY);
  appliquerStyleEntete_(sheet.getRange(1, 1, 1, columnCount));
}

function appliquerMiseEnFormeConditionnelleHorodateurs_(sheet, columnCount) {
  var dataRange = sheet.getRange(2, 1, Math.max(sheet.getMaxRows() - 1, 1), columnCount);
  var rules = sheet.getConditionalFormatRules().filter(function(rule) {
    var ranges = rule.getRanges();
    var i = 0;

    for (i = 0; i < ranges.length; i += 1) {
      if (ranges[i].getSheet().getSheetId() === sheet.getSheetId()) {
        return false;
      }
    }

    return true;
  });

  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=$E2=TRUE')
      .setBackground('#d9ead3')
      .setRanges([dataRange])
      .build()
  );

  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=OR($C2="",$D2="")')
      .setBackground('#f4cccc')
      .setRanges([dataRange])
      .build()
  );

  sheet.setConditionalFormatRules(rules);
}

function creerOuRemplacerFiltre_(sheet, columnCount) {
  var filter = sheet.getFilter();
  var rowCount = Math.max(sheet.getLastRow(), 2);

  if (filter) {
    filter.remove();
  }

  sheet.getRange(1, 1, rowCount, columnCount).createFilter();
}

function assurerDimensionsMinimales_(sheet, minRows, minColumns) {
  if (sheet.getMaxRows() < minRows) {
    sheet.insertRowsAfter(sheet.getMaxRows(), minRows - sheet.getMaxRows());
  }

  if (sheet.getMaxColumns() < minColumns) {
    sheet.insertColumnsAfter(sheet.getMaxColumns(), minColumns - sheet.getMaxColumns());
  }
}
