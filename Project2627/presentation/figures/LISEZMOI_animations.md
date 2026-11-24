# Animations du modèle de voiture à 4 DDL — choix de représentation

Document généré automatiquement par `make_animations.py` : les valeurs ci-dessous
correspondent exactement aux vidéos du dossier. Ne pas le modifier à la main
(il est réécrit à chaque génération).

**Commun à toutes les animations**

- Le dessin est un schéma : longueurs horizontales à l'échelle (empattement L = 4 m,
  roues à ±2 m du centre de gravité), hauteurs des blocs et ressorts arbitraires.
- Les déplacements verticaux réels (mm, cm) seraient invisibles : ils sont **amplifiés**
  d'un facteur indiqué pour chaque animation. « Amplifié ×N » signifie qu'un
  déplacement réel de 1 cm est dessiné comme N cm (à la même échelle que l'empattement).
- Rectangles en pointillés : position d'équilibre. Flèches ↕ : déplacements imposés
  par la route x01 (roue 1, arrière, à gauche) et x02 (roue 2, avant, à droite).
- Pneus : ressort k et amortisseur b ; suspensions : ressort K0 et amortisseur B0.
- Aucun texte sur les images : les informations sont dans les noms de fichiers.

<!-- debut:modes -->
## 1_modes — modes propres

Fichiers `mode<n>_<type>_f<fréquence>Hz.mp4` : vibration libre dans le mode n, en boucle (2 périodes, raccord parfait). La route est immobile.

- Un mode propre n'a pas d'amplitude physique (vecteur propre défini à une constante près) : chaque mode est mis à l'échelle pour que son plus grand déplacement soit bien visible. Seules les **proportions entre DDL** ont un sens ; elles sont données ci-dessous.
- Signe arbitraire : le plus grand déplacement part vers le haut.
- Une période dure 2 s à l'écran : l'animation est **ralentie** (facteur ci-dessous).
- Aperçu statique : `modes_apercu_...png/.svg`, position extrême de chaque mode (ordre : 1 en haut à gauche, 2 en haut à droite, 3 en bas à gauche, 4 en bas à droite).

| Fichier | f [Hz] | Ralenti | x1 | x2 | caisse au-dessus roue 1 | caisse au-dessus roue 2 |
|---|---|---|---|---|---|---|
| `mode1_pompage-de-la-caisse_f1.15Hz.mp4` | 1.15 | ×2.3 | +0.40 | +0.40 | +1.00 | +1.00 |
| `mode2_tangage-de-la-caisse_f1.18Hz.mp4` | 1.18 | ×2.4 | -0.40 | +0.40 | -1.00 | +1.00 |
| `mode3_roues-en-phase_f20.47Hz.mp4` | 20.47 | ×40.9 | +1.00 | +1.00 | -0.01 | -0.01 |
| `mode4_roues-en-opposition_f20.47Hz.mp4` | 20.47 | ×40.9 | +1.00 | -1.00 | -0.01 | +0.01 |

Proportions normalisées : 1,00 = plus grand déplacement du mode ; signe opposé = mouvement en opposition de phase.
<!-- fin:modes -->

<!-- debut:transmissibilite -->
## 2_transmissibilite — réponse harmonique près de la 1re résonance

Fichiers `<cas>_<transmissibilité>_f<fréquence>Hz_gain<valeur>_route-dessinee-x<N>.mp4` : réponse harmonique établie du **modèle complet** (force non simplifiée), en boucle (2 périodes, raccord parfait).

- Fréquence : la valeur de la grille `f` du notebook juste en dessous de la première résonance de la courbe (la résonance exacte n'est pas sur la grille).
- `gain` = |X3/X01| (cas symétrique) ou |q3/X01| en rad/m (cas antisymétrique) à cette fréquence.
- Route et voiture n'ont **pas la même amplification** : sinon, la route serait invisible à côté de la caisse. `route-dessinee-xN` : la route est dessinée N fois plus grande que si elle avait l'échelle de la voiture.
- Une période dure 2 s à l'écran (ralenti indiqué ci-dessous).

### `X01=X02_symetrique_X3-sur-X01_f1.10Hz_gain11.8_route-dessinee-x6.mp4`

- Excitation : X01 = X02 (même déplacement sous les deux roues), f = 1.102 Hz (résonance du mode 1 : 1.152 Hz) ; ralenti ×2.2.
- Pour 1 mm d'amplitude de route : roue 1 5.26 mm, roue 2 5.26 mm, caisse 11.78 mm, rotation 0° (soit 0.00 mm aux points d'attache).
- La réponse est linéaire : l'amplitude de la route est arbitraire, seules comptent les proportions ci-dessus. À l'écran, la route est dessinée 6 fois plus grande qu'à l'échelle de la voiture.

### `X01=-X02_antisymetrique_q3-sur-X01_f1.10Hz_gain3.84rad-par-m_route-dessinee-x4.mp4`

- Excitation : X01 = −X02 (roues excitées en opposition de phase), f = 1.102 Hz (résonance du mode 2 : 1.182 Hz) ; ralenti ×2.2.
- Pour 1 mm d'amplitude de route : roue 1 3.64 mm, roue 2 3.64 mm, caisse 0.00 mm, rotation 0.22° (soit 7.67 mm aux points d'attache).
- La réponse est linéaire : l'amplitude de la route est arbitraire, seules comptent les proportions ci-dessus. À l'écran, la route est dessinée 4 fois plus grande qu'à l'échelle de la voiture.
<!-- fin:transmissibilite -->

<!-- debut:temps -->
## 3_reponse_temporelle — réponse au profil de route

Fichiers `reponse-route_v<vitesse>ms_<lecture>_deplacements-x<N>_duree<T>s.mp4` : réponse temporelle calculée dans le notebook (convolution dans la base modale) sur le profil `w_road`, une vidéo par vitesse.

- Lecture : temps réel (option `--speed` du script).
- Voiture **et** route sont amplifiées du **même** facteur N (`deplacements-xN`) : les rapports d'amplitude entre route, roues et caisse sont respectés. N est choisi pour que le plus grand déplacement de la vidéo soit bien visible, il change donc avec la vitesse : comparer les vidéos entre elles se fait avec les valeurs ci-dessous, pas à l'œil.
- Panneau du bas : profil `w_road` réel (non amplifié par rapport à lui-même, axe vertical en mm, de -16.2 à 14.5 mm) sur ±3,6 m autour du centre de la voiture ; il défile à la vitesse du véhicule (marche vers la droite). Les points marquent le contact des roues, les traits pointillés sont alignés sous les roues du dessin.

| Fichier | v [m/s] | Amplification | max roue 1 | max roue 2 | max caisse x3 | max rotation q3 | max route |
|---|---|---|---|---|---|---|---|
| `reponse-route_v05ms_temps-reel_deplacements-x5_duree40s.mp4` | 5 | ×5 | 3.00 cm | 3.46 cm | 6.09 cm | 0.577° | 1.62 cm |
| `reponse-route_v10ms_temps-reel_deplacements-x9_duree20s.mp4` | 10 | ×9 | 2.28 cm | 2.37 cm | 1.04 cm | 1.191° | 1.62 cm |
| `reponse-route_v30ms_temps-reel_deplacements-x25_duree7s.mp4` | 30 | ×25 | 1.63 cm | 1.49 cm | 2.21 mm | 0.124° | 1.62 cm |
<!-- fin:temps -->
