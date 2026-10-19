# M2 TSI – IA : Guide d'étude complet (Séances 1 → 4)

> Objectif : remplacer les quatre notebooks. Chaque bloc de code est expliqué, chaque question 🏫 est corrigée, avec les plus petits exemples possibles.
> Les valeurs marquées **(≈)** sont des résultats typiques : elles dépendent de la machine et de la graine, donc dis « environ » à l'examen. Les valeurs sans ≈ sont exactes (nombres de paramètres, formes, nombres de mises à jour).

---

## 0. Vue d'ensemble : de quoi parlent les 4 séances

| Séance | Base de données | Thème | Compétence clé |
|---|---|---|---|
| 1 | MNIST (chiffres) | Premier réseau dense | Format des données, normalisation, one-hot, activations, compile/fit, erreurs |
| 2 | Fashion-MNIST | Hyperparamètres | Expériences équitables, taux d'apprentissage, taille des lots, largeur, activations, époques, validation vs test |
| 3 | Fashion-MNIST (500 images) | Surapprentissage + CNN | Dropout, L2, arrêt anticipé, convolution, Conv2D/Pooling, comptage de paramètres |
| 4 | MNIST | Robustesse + génération | Matrice de confusion, bruit/translation, augmentation de données, F = N∘G |

**Le pipeline à savoir réciter** (squelette de tous les notebooks) :

1. Charger les données → 2. Normaliser les pixels dans [0,1] → 3. One-hot des étiquettes → 4. Construire le modèle (`Sequential`) → 5. `compile` (optimiseur, perte, métriques) → 6. `fit` (époques, taille de lot, validation) → 7. Lire les courbes (loss/accuracy, entraînement vs validation) → 8. Choisir le modèle avec la **validation** → 9. Évaluer **une seule fois** sur le **test**.

---

## 1. Les bases (à lire d'abord, tout le reste en dépend)

### 1.1 Une image est un tableau de nombres
Une image 28×28 en niveaux de gris est une matrice de 28 lignes et 28 colonnes. Chaque case (pixel) est un entier de 0 (noir) à 255 (blanc).

Mini-exemple (« image » 3×3) :
```
[[  0, 255,   0],
 [  0, 255,   0],
 [  0, 255,   0]]   → une ligne blanche verticale
```
`img[1, 2]` = ligne 1, colonne 2 = 0. **L'indexation commence à 0, d'abord les lignes, puis les colonnes.**

### 1.2 Les formes (shapes) : ce que les examinateurs adorent
| Objet | Forme | Signification |
|---|---|---|
| `x_train` MNIST | `(60000, 28, 28)` | 60000 images de 28×28 |
| `y_train` MNIST | `(60000,)` | 60000 étiquettes entières |
| `x_test` | `(10000, 28, 28)` | |
| `y_test` | `(10000,)` | |
| Après `[..., None]` | `(N, 28, 28, 1)` | dernier axe = **canaux** (1 = gris, 3 = RVB) |
| Après `to_categorical` | `(N, 10)` | one-hot |

Le premier axe est toujours l'**axe des échantillons / du lot (batch)** : N images.

### 1.3 Un neurone et une couche dense
Un neurone calcule : `sortie = activation(w1·x1 + w2·x2 + ... + b)`.
- `w` = **poids**, `b` = **biais**. Les deux sont appris.
- Une couche `Dense(n)` avec `p` entrées a `p·n` poids + `n` biais = **`p·n + n` paramètres**.

Exemple : `Dense(3)` alimentée par 2 nombres → 2·3 + 3 = 9 paramètres.

### 1.4 Paramètres vs hyperparamètres
- **Paramètres** : appris par l'entraînement (poids, biais).
- **Hyperparamètres** : choisis par toi avant l'entraînement : taux d'apprentissage, taille des lots, nombre de neurones, nombre d'époques, activation, taux de dropout, optimiseur, architecture, coefficient L2.

### 1.5 Comment fonctionne l'entraînement (la boucle en mots simples)
1. Prendre un **lot (batch)** (petit groupe d'images, ex. 128).
2. **Propagation avant** : calculer les prédictions.
3. Calculer la **perte (loss)** (à quel point on se trompe).
4. **Rétropropagation** : calculer le gradient (dérivée de la perte par rapport à chaque poids) avec la règle de dérivation en chaîne.
5. **Mise à jour par l'optimiseur** : on ajuste chaque poids pour réduire la perte.
6. Répéter pour chaque lot. Quand toutes les données d'entraînement ont été vues = **1 époque**.

Mini-exemple de descente de gradient : perte = w², départ w = 1, taux d'apprentissage 0,1. Gradient = 2w = 2. Nouveau w = 1 − 0,1·2 = 0,8. Ensuite : gradient 1,6 → w = 0,64. On glisse vers 0. Avec un taux de 1,5 : w = 1 − 3 = −2 → puis 4 → puis −8 : ça **diverge**. C'est pourquoi un taux trop grand échoue.

### 1.6 Surapprentissage vs sous-apprentissage
- **Sous-apprentissage** : mauvais sur l'entraînement ET la validation (modèle trop faible ou trop peu entraîné).
- **Surapprentissage** : la perte d'entraînement continue de baisser mais la **perte de validation remonte** (le modèle mémorise les particularités de l'entraînement).
- Un simple écart entre entraînement et validation **ne suffit pas** à prouver le surapprentissage ; il faut voir la perte de validation remonter de façon persistante pendant que la perte d'entraînement baisse (les notebooks le répètent souvent).

### 1.7 Les trois jeux de données
| Jeu | Rôle | Règle |
|---|---|---|
| Entraînement | ajuster les poids | |
| Validation | comparer les hyperparamètres / choisir les époques / choisir le modèle | réutilisable plusieurs fois |
| Test | estimation finale honnête | **à utiliser UNE fois à la fin ; ne jamais régler à partir de lui** (sinon il n'est plus indépendant : biais optimiste, « fuite de données ») |

---

## 2. SÉANCE 1 : MNIST et un premier réseau de neurones

### 2.1 Imports
```python
from tensorflow import keras        # la bibliothèque de deep learning (Keras sur TensorFlow)
import matplotlib.pyplot as plt     # tracés
import numpy as np                  # tableaux / maths
```

### 2.2 Charger MNIST
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```
MNIST = chiffres manuscrits 0–9, niveaux de gris 28×28. Renvoie deux tuples : (images d'entraînement, étiquettes d'entraînement), (images de test, étiquettes de test). Le type des pixels est `uint8` (0–255).

### 2.3 Format des données
```python
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 28, 28)
print(y_test.shape)   # (10000,)
```

**🏫 Que représente `x_train[17]` ?** La 18ᵉ image d'entraînement (l'indice 0 = la première), une matrice 28×28 d'entiers de 0 à 255.
**🏫 Que représente `y_train[17]` ?** Son étiquette : le chiffre représenté (entier 0–9). Dans le MNIST standard c'est 8 (exécute `y_train[17]` pour confirmer).
**🏫 Que représente `x_train[17,12,4]` ?** Le pixel de l'image 17, **ligne 12, colonne 4** : un entier 0–255 (luminosité). La colonne 4 est proche du bord gauche, donc très probablement 0 (fond noir). La question teste que tu connais l'ordre : image, ligne, colonne.

### 2.4 Afficher une image
```python
def affiche_image(x):
    plt.imshow(x, cmap="gray")   # dessine la matrice en niveaux de gris
    plt.colorbar()               # légende : couleur ↔ valeur
    plt.show()                   # affiche
```
- **🏫 `affiche_image(x_train[43])`** : affiche la 44ᵉ image (indice 43). La barre de couleur va de 0 à 255.
- **🏫 `y_train[43]`** : le chiffre affiché. Exécute-le ; il doit correspondre à ce que tu vois (bon contrôle de cohérence images/étiquettes).

### 2.5 Prétraitement de x (les images)
```python
print(x_train.min())  # 0
print(x_train.max())  # 255
x_train = x_train.astype("float32") / x_train.max()   # maintenant dans [0,1]
x_test  = x_test.astype("float32") / y_test.max()     # ⚠️ BUG du notebook
```
**Pourquoi normaliser ?** Les réseaux s'entraînent mieux et plus vite avec des entrées petites et d'échelle similaire : les gradients sont bien conditionnés et les poids initiaux sont adaptés à de petites entrées. [0,1] est le standard.

**⚠️ Piège n°1 :** la deuxième ligne divise `x_test` par **`y_test.max()`**, qui à ce moment-là est la plus grande *étiquette* = **9**, pas 255 ! Les pixels de test seraient dans [0 ; 28,3] au lieu de [0,1], alors que le réseau a appris sur [0,1]. L'accuracy sur le test chuterait fortement. **Correct :** `x_test = x_test.astype("float32") / 255.0` (ou `/ x_train.max()`). Règles : (1) diviser le test par la même constante que l'entraînement, (2) une *étiquette* ne sert jamais à normaliser des *images*. Le markdown dit même « x_train et y_test » (coquille, même bug).

**🏫 À quoi sert `astype("float32")` ?** Les entiers `uint8` ne peuvent pas stocker des fractions comme 0,37 (et l'arithmétique `uint8` peut déborder). `float32` = nombres réels sur 32 bits : c'est le type par défaut de Keras/TensorFlow, deux fois moins de mémoire que float64, et rapide.

### 2.6 Prétraitement de y (les étiquettes) : encodage one-hot
```python
y_train[43]                                   # ex. 5  (un entier)
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)
y_train[43]                                   # array([0,0,0,0,0,1,0,0,0,0])
```
**🏫 Que fait `to_categorical` ?** Un **encodage one-hot** : l'entier `k` devient un vecteur de longueur 10 avec un 1 en position k et 0 ailleurs. Exemples : `3 → [0,0,0,1,0,0,0,0,0,0]`, `0 → [1,0,0,0,0,0,0,0,0,0]`.
**Pourquoi ?** Le réseau sort 10 probabilités (une par chiffre) ; `categorical_crossentropy` compare ce vecteur de probabilités à une cible de même taille. (Alternative : garder les entiers et utiliser `sparse_categorical_crossentropy`.)

**🏫 Comment retrouver le chiffre ?** La position du 1 = `np.argmax` :
```python
def chiffre(y):
    return int(np.argmax(y))      # chiffre([0,0,0,0,0,0,0,1,0,0]) → 7
```
`argmax` renvoie l'**indice** de la plus grande valeur (pas la valeur).

### 2.7 Fonctions d'activation
Activation = fonction non linéaire appliquée à la sortie de chaque neurone. Sans elle, empiler des couches revient à une seule transformation linéaire (la profondeur ne sert à rien).

| Nom | Formule | Image | Forme / usage |
|---|---|---|---|
| **sigmoïde** | σ(x) = 1 / (1 + e^(−x)) | (0,1) | courbe en S ; σ(0)=0,5 ; sature (plate) pour \|x\| grand → gradients minuscules (gradient évanescent) |
| **ReLU** | max(0, x) | [0, ∞) | 0 pour les négatifs, identité pour les positifs ; peu coûteuse, choix par défaut en couche cachée |
| **softmax** | softmax(x)ᵢ = e^(xᵢ) / Σⱼ e^(xⱼ) | chaque valeur dans (0,1), somme = 1 | Transforme un vecteur de scores en **distribution de probabilités** ; utilisée en **sortie** pour le multi-classe |

Exemples numériques :
- sigmoïde(0)=0,5 ; sigmoïde(2)≈0,88 ; sigmoïde(−20)≈0.
- relu(−3)=0 ; relu(2,5)=2,5.
- softmax([2,1,0]) : e²=7,389, e¹=2,718, e⁰=1 → somme=11,107 → **[0,665 ; 0,245 ; 0,090]**. Le plus grand score reçoit la plus grande probabilité ; la somme fait 1.

Le code de tracé :
```python
x = np.arange(-20.0, 20.0, 0.1)                      # points de −20 à 20 par pas de 0,1
y_sigmoid = keras.activations.sigmoid(x)
fig, ax = plt.subplots(figsize=(10, 5)); ax.axvline(color='gray'); ax.axhline(color='gray'); ax.plot(x, y_sigmoid)
```
`axvline/axhline` tracent les axes x=0 et y=0 en gris ; `plot(x, y)` trace la courbe.

### 2.8 Construire le réseau
```python
model = keras.Sequential([
    keras.layers.Input(shape=(28,28)),              # indique la forme d'entrée
    keras.layers.Flatten(),                         # (28,28) → (784,)
    keras.layers.Dense(128, activation="relu"),     # couche cachée, 128 neurones
    keras.layers.Dense(10, activation="softmax")    # sortie : 10 probabilités
])
```
`Sequential` = couches empilées les unes après les autres.

**🏫 À quoi sert `Flatten` ?** Une couche dense attend un vecteur plat, pas une grille 2D. `Flatten` déroule l'image 28×28 ligne par ligne en 784 nombres (28·28=784). Elle a **0 paramètre** (simple remise en forme). Inconvénient : la structure de voisinage 2D est perdue (les CNN de la séance 3 règlent ça).

**`model.summary()`** affiche les couches, formes de sortie, nombres de paramètres :
| Couche | Forme de sortie | Paramètres |
|---|---|---|
| Flatten | (None, 784) | 0 |
| Dense(128) | (None, 128) | 784·128 + 128 = **100 480** |
| Dense(10) | (None, 10) | 128·10 + 10 = **1 290** |
| **Total** | | **101 770** |

(`None` = taille du lot, non fixée.)

### 2.9 Compile
```python
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```
- **🏫 `optimizer="adam"`** : l'algorithme qui met à jour les poids à partir des gradients. **Adam** = variante de la descente de gradient avec **moment** (direction du gradient lissée) et **pas adaptatif par poids**. Défaut robuste, demande peu de réglage (taux par défaut 0,001).
- **🏫 `loss="categorical_crossentropy"`** : la quantité à **minimiser**. Pour un échantillon de cible one-hot y et de probabilités prédites p : **L = −Σₖ yₖ log pₖ = −log(p de la vraie classe)**. Exemples : p_vraie = 0,9 → L=0,105 ; 0,5 → 0,69 ; 0,1 → 2,30 ; 0,01 → 4,6. Elle punit fortement les mauvaises réponses données avec confiance, et elle est dérivable (l'accuracy ne l'est pas, c'est pourquoi on ne l'optimise pas directement).
- **🏫 `metrics=["accuracy"]`** : mesure supplémentaire pour la lecture seulement (non optimisée) : proportion d'échantillons bien classés. Exemple : 97 bons sur 100 → 0,97.

### 2.10 Fit
```python
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
```
(La cellule « raw » contient la coquille `hystory` ; la cellule exécutée utilise `history`.)
- **🏫 `epochs=10`** : 10 passages complets sur les données d'entraînement.
- **🏫 `batch_size=128`** : les poids sont mis à jour tous les 128 échantillons (mini-lot). Petits lots → plus de mises à jour, gradients plus bruités ; grands lots → moins de mises à jour, plus lisses, plus rapide par époque.
- **🏫 `validation_split=0.2`** : Keras met de côté les **20 % derniers** des données d'entraînement fournies (pris *avant* le mélange) et ne s'entraîne pas dessus ; il mesure seulement perte/accuracy dessus après chaque époque. → 48 000 entraînement / 12 000 validation. Étapes par époque : 48000/128 = **375**.

Sortie par époque : `loss`, `accuracy`, `val_loss`, `val_accuracy` :
- **🏫 `accuracy`** : proportion de prédictions correctes sur les données d'**entraînement** (moyenne pendant l'époque, alors que les poids changent).
- **🏫 `loss`** : entropie croisée moyenne sur l'entraînement.
- **🏫 `val_accuracy`** : accuracy sur les données de **validation** mises de côté (jamais utilisées pour apprendre).
- **🏫 `val_loss`** : entropie croisée sur la validation.
- **🏫 Laquelle est la plus importante ?** Celles de **validation** (`val_accuracy`, et `val_loss`) : elles estiment la généralisation à des données inédites, qui est le vrai but. Les métriques d'entraînement disent seulement à quel point on mémorise. Résultat typique **(≈)** : acc. entraînement ≈ 0,99, acc. validation ≈ 0,97–0,98.

### 2.11 Courbes d'apprentissage
```python
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()
```
`history.history` est un dictionnaire : clés `loss, accuracy, val_loss, val_accuracy`, chacune une liste avec une valeur par époque. Dans la cellule d'accuracy, la dernière ligne est juste `plt` (il manque `.show()` ; petit bug : rien n'est affiché explicitement).

**🏫 Pourquoi la courbe bleue passe-t-elle au-dessus de l'orange (courbe d'accuracy) ?** La première couleur par défaut de Matplotlib (bleu) = « train », la deuxième (orange) = « validation ». L'accuracy d'entraînement est généralement plus haute car les poids ont été ajustés sur ces images exactes ; les images de validation sont nouvelles. (Sur la courbe de loss, c'est l'inverse : loss d'entraînement plus basse.) De plus, les métriques d'entraînement sont moyennées pendant l'époque alors que le modèle progresse encore, ce qui peut inverser les choses aux premières époques.

### 2.12 Analyse des erreurs
```python
predictions = model.predict(x_test)                 # forme (10000, 10) : probabilités
erreurs = [i for i in range(len(predictions))
           if np.argmax(predictions[i]) != np.argmax(y_test[i])]
len(erreurs)                                         # ≈ 200–250 mal classées sur 10000
```
Pour chaque image de test i : chiffre prédit = argmax des probabilités prédites ; vrai chiffre = argmax du one-hot. S'ils diffèrent → erreur. (Avec le bug `/ y_test.max()`, ce nombre serait bien plus grand.)
```python
def affiche_erreur(e):
    y  = np.argmax(y_test[e])
    yp = np.argmax(predictions[e])
    print("Le réseau de neurone identifie un", yp, "au lieu d'un", y)
    affiche_image(x_test[e])
```
Affiche « le réseau voit un *yp* au lieu d'un *y* », puis l'image. `for e in erreurs[0:10]: affiche_erreur(e)` montre les dix premières. Observation : les erreurs sont souvent des chiffres ambigus/mal écrits sur lesquels un humain hésiterait aussi.

---

## 3. SÉANCE 2 : Fashion-MNIST et hyperparamètres

### 3.1 Mise en place
```python
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter       # pour mesurer les durées

GRAINE = 42
keras.utils.set_random_seed(GRAINE) # fixe les graines Python/NumPy/TF → reproductibilité
```
Une **graine** rend l'initialisation et le mélange « aléatoires » répétables. Les résultats peuvent quand même varier d'une machine à l'autre.

### 3.2 Les données
```python
(x_train_brut, y_train_brut), (x_test_brut, y_test_brut) = keras.datasets.fashion_mnist.load_data()
noms_classes = ["T-shirt / haut","Pantalon","Pull","Robe","Manteau","Sandale","Chemise","Basket","Sac","Bottine"]
```
Fashion-MNIST : 60 000 entraînement + 10 000 test, 28×28 gris, **10 classes de vêtements** (les étiquettes 0–9 indexent `noms_classes`). Plus dur que MNIST (Chemise/T-shirt/Pull/Manteau se ressemblent). (Le notebook affiche deux fois les formes d'entraînement ; le second print était destiné au test.)

**🏫 `x_train_brut[43]`, `y_train_brut[43]`, `x_train_brut[43,12,4]` :** l'image n°43 (28×28), son numéro de classe, et la valeur 0–255 du pixel (ligne 12, colonne 4). Pour afficher le nom : `print(noms_classes[y_train_brut[43]])`. Réponse à « toutes les catégories sont-elles faciles à distinguer ? » : **non** ; T-shirt/Chemise/Pull/Manteau et Sandale/Basket/Bottine sont proches.

### 3.3 Découpage entraînement / validation / test
```python
N_TRAIN = 10000
N_VAL = 2000
indices = np.random.default_rng(GRAINE).permutation(len(x_train_brut))   # indices 0..59999 mélangés
i_train = indices[:N_TRAIN]                  # les 10000 premiers indices mélangés
i_val   = indices[N_TRAIN:N_TRAIN + N_VAL]   # les 2000 suivants

x_train = x_train_brut[i_train].astype("float32") / 255.0   # ici 255 est correct
x_val   = x_train_brut[i_val].astype("float32") / 255.0
x_test  = x_test_brut.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train_brut[i_train], 10)
...
```
- `permutation` = mélange aléatoire des indices → sous-ensembles aléatoires, à peu près équilibrés en classes, fixés par la graine (identiques pour toutes les expériences → **comparaison équitable**).
- `x[tableau_d_indices]` est l'« indexation avancée » : récupère ces lignes.
- Seuls 10 000 + 2 000 sont utilisés (le reste est ignoré) pour gagner du temps.
- La constante explicite 255 (valeur maximale possible d'un pixel) est utilisée ici, ce qui évite le bug de la séance 1.

### 3.4 Réseau de référence
**Réglages de référence :** 128 neurones, ReLU, Adam, taux d'apprentissage 0,001, lots de 128, 8 époques.

Fonction complétée (le « À COMPLÉTER ») :
```python
def cree_modele(neurones=128, taux=0.001, activation="relu"):
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(neurones, activation=activation),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=taux),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model
```
Les arguments par défaut font que `cree_modele()` donne la référence.

**🏫 Quels réglages sont des hyperparamètres ? Combien de paramètres ?** Hyperparamètres : nombre de neurones (128), activation, taux d'apprentissage, taille des lots, époques, optimiseur (et l'architecture elle-même). Paramètres appris : **101 770** :
- cachée : 784·128 poids + 128 biais = 100 352 + 128 = 100 480
- sortie : 128·10 + 10 = 1 290
- 100 480 + 1 290 = **101 770**.

### 3.5 Les outils d'expérience (code fourni, à comprendre)
```python
experiences = {}                        # dictionnaire qui stocke chaque essai

def experience(nom, neurones=128, taux=0.001, batch_size=128, epochs=8, activation="relu", graine=GRAINE):
    keras.backend.clear_session()       # libère l'ancien graphe/la mémoire
    keras.utils.set_random_seed(graine) # même aléa à chaque essai
    model = cree_modele(...)            # NOUVEAU réseau + NOUVEL optimiseur à chaque appel
    debut = perf_counter()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
    duree = perf_counter() - debut
    ...stocke modèle, historique, durée, config dans experiences[nom]...
```
Points clés :
- **Un modèle neuf à chaque appel** : ne jamais poursuivre l'entraînement d'un ancien modèle pour comparer des réglages (il aurait plus d'époques : injuste).
- `validation_data=(x_val, y_val)` : jeu de validation explicite (vs `validation_split` en séance 1).
- `shuffle=True` : remélange les échantillons d'entraînement à chaque époque.
- **Un seul hyperparamètre modifié à la fois** (tout le reste = référence) : c'est une **expérience contrôlée** (« conduire une comparaison équitable »).
- De petites différences peuvent n'être que du bruit : idéalement, répéter avec plusieurs graines.
- `courbes(history)` : 2 graphiques (loss ; accuracy) entraînement & validation. `compare(noms)` : superpose les courbes de validation de plusieurs essais. `bilan(noms)` : tableau acc. train finale, acc. val, loss val, époque du minimum de val_loss (`np.argmin(...)+1`), durée. L'astérisque `*` : les mesures sont à la **dernière** époque et le modèle stocké est celui de la dernière époque.

**🏫 Lire les courbes de référence.**
- `loss`/`accuracy` : sur les données d'entraînement. `val_loss`/`val_accuracy` : sur la validation (inédites).
- Progresse-t-il encore à l'époque 8 ? Typiquement la loss d'entraînement baisse encore et l'acc. val monte lentement ou se stabilise (≈ 0,83–0,86 d'acc. val avec 10 000 images). Dis ce que montre ton graphique.
- Un écart train/val prouve-t-il le surapprentissage ? **Non.** Un écart est normal (on a ajusté sur l'entraînement). Le surapprentissage demande une **remontée persistante de la loss val pendant que la loss train baisse**.
- Gestion du temps : si > 1 minute, réduire `N_TRAIN` à 5000 et relancer, puis le garder pour toutes les comparaisons (ne jamais changer la quantité de données en cours d'étude).

### 3.6 Section 3 : taux d'apprentissage (η)
Mise à jour des poids ≈ `w ← w − η × (direction issue du gradient)`. η contrôle la **taille du pas**.

Comparer **1e-5, 1e-3 (référence), 0,1**.

**🏫 Prédictions avant de lancer :**
- Très faible (0,00001) : pas minuscules → la loss baisse très lentement ; courbes presque plates / progression lente ; après 8 époques encore **sous-entraîné** (accuracy basse). Ce n'est pas « faux », juste trop lent pour notre budget.
- Très élevé (0,1 avec Adam) : pas trop grands → dépassement ; bruité, sauts, possiblement bloqué ou divergent ; accuracy mauvaise/instable, val_loss erratique.
- 0,001 : lisse, bon.

Code à compléter :
```python
experience("taux faible", taux=0.00001)
experience("taux élevé",  taux=0.1)
compare(["taux faible", "référence", "taux élevé"]); bilan([...])
```
**🏫 Observations (≈) :** référence acc. val ≈ 0,84–0,86 ; taux faible nettement plus bas (≈ 0,6–0,75) et encore en hausse à l'époque 8 (**trop lent pour le budget, pas forcément mauvais à long terme**) ; taux élevé bruité/instable et moins bon (≈ 0,7–0,8, peut fluctuer fortement). Conclusion : le taux d'apprentissage a un **optimum** ; « plus grand ≠ meilleur », « plus petit ≠ plus sûr ».

### 3.7 Section 4A : taille des lots
Comparer **32, 128 (réf.), 512**, même η = 0,001 (un seul changement).

**🏫 Mises à jour par époque** = ⌈N_TRAIN / batch_size⌉ :
- 32 → ⌈10000/32⌉ = ⌈312,5⌉ = **313**
- 128 → ⌈78,125⌉ = **79**
- 512 → ⌈19,53⌉ = **20**
(×8 époques : 2504, 632, 160 mises à jour au total.)

```python
experience("lot 32",  batch_size=32)
experience("lot 512", batch_size=512)
```
**🏫 Compromis (≈) :** petit lot = plus de mises à jour, donc progression plus rapide par époque et souvent meilleurs résultats val à nombre d'époques fixé, mais **temps réel plus long** (plus d'étapes, parallélisme moins bien exploité) et gradients plus bruités. Grand lot = rapide par époque, gradients stables, mais moins de mises à jour → **sous-entraîné** après 8 époques. « Même nombre d'époques » ≠ « même nombre de mises à jour » (la taille de lot change leur nombre) ≠ « même temps de calcul » (temps par étape et efficacité du parallélisme diffèrent).

### 3.8 Section 4B : nombre de neurones (capacité)
Comparer **16, 128, 512** neurones.
| Neurones | Paramètres |
|---|---|
| 16 | 784·16+16 = 12 560 ; 16·10+10 = 170 ; **12 730** |
| 128 | **101 770** |
| 512 | 784·512+512 = 401 920 ; 512·10+10 = 5 130 ; **407 050** |

```python
experience("16 neurones",  neurones=16)
experience("512 neurones", neurones=512)
for nom in ["16 neurones","référence","512 neurones"]:
    print(nom, experiences[nom]["model"].count_params())
```
`count_params()` renvoie le total de paramètres.
**🏫 Plus large est-il toujours meilleur ?** Non. 16 neurones : sous-apprentissage (capacité insuffisante). 128→512 : gain faible ou nul en validation (≈ +0–1 pt), 4× plus de paramètres, plus de temps. On ne conclut au surapprentissage que si la loss val remonte pendant que la loss train baisse ; après 8 époques ce n'est pas encore établi.

### 3.9 Section 5 : fonctions d'activation
Activations comparées pour la couche cachée : softmax (comme dit le texte ; les commentaires du code disent « sigmoid » !), relu (réf.), tanh, leaky_relu.
- **tanh(x)** = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ), image (−1, 1), centrée en zéro, sature.
- **Leaky ReLU** = x si x>0 sinon α·x (petite pente α≈0,2–0,3 dans Keras) ; évite les « neurones morts » (ReLU donne un gradient exactement nul pour les négatifs).
- **Softmax en couche cachée** : mauvais choix : elle force les 128 sorties à sommer à 1 (compétition/goulot, chaque valeur minuscule), ce qui réduit l'information. Elle est faite pour la sortie.
- **Sigmoïde** en couche cachée : fonctionne mais sature → apprentissage plus lent (gradients évanescents).

```python
experience("sigmoid",    activation="sigmoid")
experience("tanh",       activation="tanh")
experience("leaky_relu", activation="leaky_relu")
```
**⚠️ Piège n°2 :** le notebook appelle ensuite `compare(["sigmoid","référence","tanh","leaky relu"])` avec une **espace** (« leaky relu ») alors que l'essai est stocké sous « leaky_relu » → `KeyError`. Les noms doivent correspondre exactement. De plus le titre markdown dit « softmax » alors que l'essai s'appelle « sigmoid » ; pour comparer softmax, ajouter `experience("softmax", activation="softmax")`.
**🏫 Conclusion (≈) :** ReLU, leaky ReLU et tanh donnent les meilleurs résultats, similaires ; sigmoïde plus lente ; softmax cachée la pire. ReLU/leaky ReLU = les plus adaptées (simples, peu coûteuses, pas de saturation pour les positifs).

### 3.10 Section 6A : quand s'arrêter (nombre d'époques)
```python
h = history_ref.history
for e in [1, 2, 4, 8]:
    print(f"Époque {e} : train acc = {h['accuracy'][e-1]:.3f}, val acc = ..., val loss = ...")
```
`[e-1]` car les listes Python commencent à 0 (époque 1 → indice 0). `:.3f` = 3 décimales. Pas de réentraînement : l'historique contient les mesures de chaque époque.
**🏫 Quelle durée ?** Choisir l'époque où **val_loss est minimale** (ou après laquelle elle ne s'améliore plus), en vérifiant l'acc. val et le coût : par ex. si la loss val est la plus basse vers l'époque 6–8 et que l'acc. val a atteint un plateau, choisir celle-là ; moins d'époques si les gains sont négligeables.
**🏫 La loss val peut-elle monter alors que l'accuracy change peu ?** **Oui.** La loss mesure la confiance (probabilités), l'accuracy ne compte que l'argmax. Un modèle peut devenir trop confiant sur ses quelques erreurs (ex. p_vraie de 0,4 à 0,01 → loss de 0,92 à 4,6) sans que la classe prédite (donc l'accuracy) change.

### 3.10b Section 6B : tableau de synthèse (quoi écrire)
| Hyperparamètre | Valeurs | Observation val | Conclusion |
|---|---|---|---|
| Taux d'apprentissage | 1e-5 ; 1e-3 ; 0,1 | 1e-3 meilleur ; 1e-5 lent ; 0,1 instable | Optimum au milieu |
| Taille des lots | 32 ; 128 ; 512 | 32 meilleure accuracy mais plus lent ; 512 le plus rapide, le moins bon | Compromis qualité / temps |
| Neurones | 16 ; 128 ; 512 | 16 sous-apprend ; 512 ≈ 128 pour 4× de paramètres | Rendements décroissants |
| Époques | 1 ; 2 ; 4 ; 8 | loss val min vers le plateau | Arrêter où la loss val est minimale |
(À remplir avec **tes** chiffres.)

**🏫 Pourquoi combiner les meilleurs réglages pris séparément ne garantit-il pas la meilleure combinaison ?** Les hyperparamètres **interagissent** : par ex. un grand lot peut demander un taux plus grand ; la largeur interagit avec les époques et la régularisation. On a fait varier un seul réglage autour d'une seule référence, donc on n'a jamais testé de combinaisons. Et le bruit (une graine, 2000 images de validation → ±≈0,8 pt) peut tromper.

### 3.11 Sélection finale et test
```python
nom_choisi = "lot 32"            # exemple : n'importe quel nom présent dans `experiences`
epoques_choisies = 8             # entier entre 1 et 8
assert nom_choisi in experiences
config = experiences[nom_choisi]["config"].copy()
if epoques_choisies == config["epochs"]:
    model_final = experiences[nom_choisi]["model"]
else:
    config["epochs"] = epoques_choisies
    model_final, _ = experience("choix final", **config)   # réentraîne à zéro avec moins d'époques
```
`assert` s'arrête avec un message d'erreur si la condition est fausse. `**config` déballe un dictionnaire en arguments nommés. Le réentraînement est nécessaire car le modèle stocké est celui de l'époque 8.
```python
loss_test, accuracy_test = model_final.evaluate(x_test, y_test, verbose=0)
```
À faire **une seule fois** ; ne plus régler ensuite. **🏫 Test proche de la validation ?** Normalement à ≈1–2 pts ; si oui, l'estimation de validation était fiable. **Deux enseignements :** (1) un hyperparamètre à la fois + modèle neuf + mêmes données/graine = comparaison équitable ; (2) choisir sur la validation, tester une seule fois à la fin. **Une limite :** un seul essai/une seule graine, seulement 10 000 images d'entraînement, validation de 2000 images (bruit), une seule couche cachée, pas de recherche de combinaisons.

### 3.12 Section 7 (bonus) : observer le surapprentissage
Montage : seulement **300 images d'entraînement**, 3 couches cachées de 256, 40 époques, lots de 32 (donc bien plus de paramètres que de données). Image typique : loss train → ~0, acc. train → ~1,0 ; la loss val atteint un **minimum après quelques époques (≈ 5–10), puis remonte** pendant que l'acc. val stagne ≈ 0,75–0,80. Trouver l'époque avec `np.argmin(history_bonus.history["val_loss"]) + 1`. Le surapprentissage est soupçonné après cette époque. (Beaucoup de choses ont changé d'un coup par rapport à la référence, on ne peut donc pas attribuer les différences à un seul hyperparamètre.)

**Arrêt anticipé :**
```python
mesure = "val_loss"
arret = keras.callbacks.EarlyStopping(monitor=mesure, mode="min", patience=3, restore_best_weights=True)
model_bonus.set_weights(poids_initiaux)    # repart des mêmes poids initiaux
model_bonus.compile(...)                    # nouvel optimiseur (remis à zéro)
history_arret = model_bonus.fit(..., callbacks=[arret])
```
- `monitor="val_loss"`, `mode="min"` : on la veut la plus basse possible.
- `patience=3` : tolère 3 époques consécutives sans amélioration avant d'arrêter.
- `restore_best_weights=True` : à la fin, remet les poids de la **meilleure époque**.
**🏫 Empêche-t-il le réseau de commencer à surapprendre ?** Non : il laisse le surapprentissage commencer et l'**arrête/l'annule** (l'entraînement va 3 époques au-delà de la meilleure, puis on revient en arrière). **Pourquoi les mesures des poids restaurés ≠ dernière époque affichée ?** La dernière époque affichée est 3 époques après la meilleure ; les poids restaurés sont ceux de la meilleure époque, donc `evaluate` donne les valeurs de la meilleure époque. **Autres remèdes :** Dropout, régularisation L2, plus de données / augmentation, réseau plus petit.

---

## 4. SÉANCE 3 : Surapprentissage et convolutions

### 4.1 Mise en place
```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"   # choisir le backend avant d'importer keras
import tensorflow as tf
...
print("TensorFlow :", tf.__version__)
print("GPU :", tf.config.list_physical_devices("GPU"))
```
Affiche la version et les GPU disponibles (liste vide = CPU seulement).

### 4.2 Données (seulement 500 images d'entraînement, exprès)
```python
N_TRAIN, N_VAL = 500, 1500
x_train = images[i_train].astype("float32")[..., None] / 255.0
```
`[..., None]` ajoute un **nouvel axe final** : `(500,28,28)` → `(500,28,28,1)` (`...` = « tous les autres axes »). Les couches de convolution attendent **(lot, hauteur, largeur, canaux)**. Des assertions vérifient la forme et l'intervalle [0,1].
**🏫 Que représente la 4ᵉ dimension ?** L'axe des **canaux** : 1 pour le niveau de gris (3 pour RVB). **Pourquoi diviser aussi le test par 255 ?** Le réseau a appris sur des pixels dans [0,1] ; les entrées de test doivent subir la **même transformation fixe**, sinon le réseau voit des données à une autre échelle. **Pourquoi ne jamais utiliser les étiquettes pour normaliser les images ?** Les étiquettes sont les cibles ; les utiliser (ou des statistiques du test) pour transformer les entrées est une **fuite de données** et n'a aucun sens en déploiement (étiquettes inconnues).

Petit jeu d'entraînement (500) + gros modèle → idéal pour le surapprentissage.

### 4.3 Le modèle dense avec options de régularisation
```python
def cree_dense(largeur=1000, dropout=0.0, l2=0.0):
    return keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(largeur, activation="relu", kernel_regularizer=keras.regularizers.l2(l2)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(largeur, activation="relu", kernel_regularizer=keras.regularizers.l2(l2)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10, activation="softmax")
    ])
```
Les valeurs par défaut `dropout=0, l2=0` → aucun effet (le réseau de référence).
**Paramètres (référence) :** 784·1000+1000 = 785 000 ; 1000·1000+1000 = 1 001 000 ; 1000·10+10 = 10 010 ; total **1 796 010** pour seulement 500 images : énormément sur-paramétré.

### 4.4 La fonction d'expérience
```python
def experience(nom, constructeur, arret=False, epochs=MAX_EPOCHS):
    keras.utils.set_random_seed(GRAINE)
    model = constructeur()                       # constructeur = une fonction qui CONSTRUIT un modèle
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)] if arret else []
    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=64, callbacks=callbacks, verbose=2)
    mesures = model.evaluate(x_val, y_val, verbose=0, return_dict=True)   # réévalue les poids conservés
    experiences[nom] = dict(model=model, h=h.history, val=mesures, secondes=...)
```
- Passer une **fonction** (`cree_dense` ou `lambda: cree_dense(dropout=0.4)`) permet de construire un modèle **neuf** à l'intérieur, après avoir fixé la graine. Une `lambda` est une mini-fonction anonyme.
- Lots de 64, 500 images → ⌈500/64⌉ = **8 mises à jour par époque**. `MAX_EPOCHS=20`.
- Elle réévalue après le fit car avec EarlyStopping les poids conservés (les meilleurs) diffèrent de ceux de la dernière époque.
- `return_dict=True` → `{"loss":..., "accuracy":...}`.

### 4.5 Observer le surapprentissage (essai de référence)
Typique **(≈)** : acc. train → ≈ 1,00, loss train → ≈ 0 ; acc. val plafonne ≈ 0,80–0,82 ; la loss val baisse pendant ≈ 5–8 époques puis **remonte**. **🏫 Où s'arrêter ?** À l'époque de loss val minimale (`np.argmin(val_loss)+1`). Le notebook insiste sur l'honnêteté : si la validation ne se dégrade pas dans ton essai, conclure « surapprentissage non établi sur cette exécution ».
Règle de temps : si la référence dépasse 2 min, utiliser `epochs=12` ensuite.

### 4.6 Moyens de limiter le surapprentissage
| Méthode | Mécanisme |
|---|---|
| **Réduire la capacité** | moins de paramètres → moins capable de mémoriser |
| **Dropout** (taux r) | pendant l'entraînement, met à 0 une fraction r des activations au hasard (et rééchelonne les autres par 1/(1−r)) ; **inactif à l'inférence**. Évite la co-dépendance entre neurones ; comme une moyenne de nombreux sous-réseaux |
| **Régularisation L2** (weight decay) | loss += λ·Σw² → pousse les poids à rester petits → fonctions plus lisses |
| **Arrêt anticipé** | s'arrêter à la meilleure loss val, restaurer les meilleurs poids |
| **Plus de données / augmentation** | plus d'exemples (transformations qui conservent l'étiquette) |

Précautions du notebook : avec le **Dropout**, les métriques d'entraînement sont calculées avec dropout ACTIF et celles de validation sans, donc pas directement comparables (la loss train peut paraître *pire* que la loss val). Avec **L2**, la `loss` inclut la pénalité, donc comparer aussi l'accuracy.

**🏫 Essai Dropout (taux 0,4, sans arrêt anticipé) :**
```python
experience("dropout", lambda: cree_dense(dropout=0.4))
courbes("dropout")
```
Attendu **(≈)** : l'accuracy train monte plus lentement et reste sous 1 (entraînement bruité), la loss val remonte moins/plus tard, l'acc. val est similaire ou légèrement meilleure (≈ +1–3 pts) que la référence. Dis « a aidé / pas clairement » selon tes courbes.

**🏫 Dropout + arrêt anticipé :** `experience("dropout + arrêt", lambda: cree_dense(dropout=0.4), arret=True)`.
- `patience=3` : arrêt après 3 époques consécutives sans amélioration de la loss val.
- `restore_best_weights=True` : retour aux poids de la meilleure époque (sinon on garde les derniers, moins bons).
- La dernière ligne de l'historique = dernière époque (3 après la meilleure) ; `evaluate` après le fit = poids restaurés (les meilleurs) → ils diffèrent.
- Affiche : nombre d'époques exécutées (`len(h["loss"])`) et meilleure époque (`np.argmin(h["val_loss"])+1`) ; époques exécutées = meilleure époque + 3 (si arrêt anticipé).

### 4.7 Comprendre une convolution
Un **filtre/noyau** (ici 3×3) glisse sur l'image ; à chaque position il multiplie ses poids par le voisinage 3×3 de pixels et additionne (produit scalaire) → un nombre → construit une **carte de caractéristiques (feature map)**.

```python
filtre = keras.layers.Conv2D(1, 3, padding="valid", use_bias=False)
filtre(x_train[:1])                                   # un appel pour créer les poids (« build »)
noyau = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype="float32")
filtre.set_weights([noyau.reshape(3,3,1,1)])          # (hauteur, largeur, canaux_entrée, nb_filtres)
carte = filtre(x_train[:1]).numpy()[0, ..., 0]        # échantillon 0, canal 0 → carte 2D
```
`Conv2D(1, 3, ...)` : **1 filtre**, taille de noyau **3** (soit 3×3). La forme des poids du noyau dans Keras est `(3,3,canaux_entrée,canaux_sortie) = (3,3,1,1)`.

**Exemple détaillé.** Voisinage (lignes) `[[0,0,1],[0,0,1],[0,0,1]]` × noyau `[[-1,0,1]]×3` = (0·−1+0·0+1·1)×3 = **3** (forte réponse : sombre à gauche, clair à droite). Sur un voisinage uniforme (toutes valeurs v) : −v+v par ligne = **0**. Voisinage clair à gauche, sombre à droite → **négatif**. Ce noyau calcule donc **colonne de droite moins colonne de gauche** = détecte les **bords verticaux** (changement d'intensité horizontal). (Techniquement Keras calcule une corrélation croisée, sans retourner le noyau.)

- **🏫 Taille du filtre ?** 3×3 (9 poids ; +1 biais s'il y en a un).
- **🏫 `padding="valid"` ?** Pas de remplissage : le filtre n'est placé que là où il tient entièrement dans l'image → sortie plus petite. (`"same"` ajoute des zéros autour pour que la sortie ait la **même hauteur/largeur** que l'entrée, avec un pas de 1.)
- **🏫 `use_bias=False` ?** Pas de constante additive b : la sortie est exactement Σ(noyau × voisinage) ; ici les 9 poids sont les seuls paramètres (le noyau fixé à la main est donc tout le filtre).
- **🏫 Forme de la carte ?** Valid : 28 − 3 + 1 = **26×26** (perd une bordure de 1 pixel de chaque côté) contre 28×28 pour l'image. Formule : sortie = (n − k + 2p)/s + 1 (p = remplissage, s = pas). Same, p=1 : (28−3+2)/1+1 = 28.

**Filtre haut/bas** (exercice) : comparer le haut et le bas :
```python
noyau_hb = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype="float32")   # ligne du bas − ligne du haut → bords horizontaux
filtre2 = keras.layers.Conv2D(1, 3, padding="valid", use_bias=False)
filtre2(x_train[:1])
filtre2.set_weights([noyau_hb.reshape(3,3,1,1)])
carte2 = filtre2(x_train[:1]).numpy()[0,...,0]
plt.imshow(carte2, cmap="coolwarm"); plt.show()
```
Il s'allume sur les **bords horizontaux** (sombre au-dessus / clair en dessous → positif).

### 4.8 Le CNN
```python
def cree_cnn():
    model = keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    assert model.output_shape[-1] == 10
    return model
```
**MaxPooling2D(2)** : découpe chaque carte en blocs 2×2, garde le **maximum** de chaque bloc → divise par deux hauteur et largeur, 0 paramètre, donne un peu de tolérance locale à la translation. Exemple : bloc `[[1,3],[2,0]]` → 3.

**🏫 Formes couche par couche :**
| Couche | Forme de sortie | Paramètres |
|---|---|---|
| Input | (28,28,1) | 0 |
| Conv2D(8,3,same) | (28,28,8) | 3·3·1·8 + 8 = **80** |
| MaxPool(2) | (14,14,8) | 0 |
| Conv2D(16,3,same) | (14,14,16) | 3·3·8·16 + 16 = **1 168** |
| MaxPool(2) | (7,7,16) | 0 |
| Flatten | (784,) [7·7·16] | 0 |
| Dense(32) | (32,) | 784·32 + 32 = **25 120** |
| Dense(10) | (10,) | 32·10 + 10 = **330** |
| **Total** | | **26 698** |

Formule des paramètres de Conv : **(k_h · k_w · C_entrée + 1) · C_sortie**.

**🏫 Comparer avec le dense de référence :** 26 698 contre 1 796 010 → environ **67× moins** de paramètres, et généralement une meilleure/égale accuracy de validation (≈ 0,85+). Pourquoi : **partage des poids** (le même filtre est réutilisé à chaque position) + connexions locales + exploitation de la structure 2D. Moins de paramètres → moins de surapprentissage, même si un CNN ne *garantit* pas l'absence de surapprentissage (il peut encore surapprendre avec 500 images ; d'où l'EarlyStopping dans `experience("CNN", cree_cnn, arret=True, epochs=12)`).

**Visualisation des cartes de caractéristiques :**
```python
extracteur = keras.Model(inputs=cnn.inputs, outputs=cnn.layers[0].output)   # modèle qui s'arrête à la 1re conv
cartes = extracteur(x_val[:1], training=False).numpy()[0]                    # (28,28,8)
```
Montre 8 cartes (une par filtre). **🏫 Identiques ?** Non : chaque filtre a des poids différents → réponses différentes (bords, taches, luminosité...). **Pourquoi ne pas donner un sens humain précis à chaque filtre ?** Les filtres sont **appris**, pas conçus ; ils dépendent de l'initialisation aléatoire/de la graine/des données, peuvent être redondants ou encoder des motifs mélangés ; l'interprétation est souvent une histoire a posteriori. Seul un vague « détecte un truc de type bord » est prudent.

### 4.9 Choix final
```python
for nom, e in experiences.items():
    print(f"{nom:20s} paramètres={e['model'].count_params():7d} acc val={e['val']['accuracy']:.3f} durée={e['secondes']:.1f}s")
```
Remplir le tableau avec tes chiffres (Référence / Dropout / Dropout + arrêt / CNN : accuracy val, paramètres, diagnostic). Diagnostics typiques : Référence : surapprend (train≈1, plateau val, 1,8M de paramètres) ; Dropout : surapprentissage réduit ; Dropout+arrêt : meilleure époque conservée, moins coûteux ; CNN : meilleure accuracy avec 26 698 paramètres.
**⚠️ Piège n°3 :** `nom_choisi =   # À COMPLÉTER` est une **SyntaxError** tant que tu n'écris pas une chaîne, par ex. `nom_choisi = "CNN"` (choisir sur la validation et le coût, puis évaluer une fois sur le test).
**🏫 Questions de clôture :**
- *Un signe de surapprentissage :* la loss train baisse pendant que la loss val monte (écart net et persistant).
- *Deux moyens de le limiter :* Dropout, arrêt anticipé (aussi L2, moindre capacité, plus de données/augmentation).
- *Intérêt du partage des poids :* beaucoup moins de paramètres, le même détecteur de motif s'applique partout (équivariance par translation), meilleure généralisation et coût réduit.
- *Pourquoi ne pas modifier le modèle après le test ?* Utiliser le résultat du test pour ajuster le modèle fait entrer le test dans la sélection du modèle → ce n'est plus une estimation indépendante (optimiste, fuite).

---

## 5. SÉANCE 4 : robustesse MNIST, matrice de confusion, génération

### 5.1 Mise en place et budgets
`N_TRAIN, N_VAL = 20000, 5000` ; `EPOCHS_N=8`, `BATCH_SIZE=128`, `EPOCHS_G=15`, `N_LATENT_TRAIN=4096`. Machine lente : `N_TRAIN=10000`, `EPOCHS_N=5`, `EPOCHS_G=8`.

### 5.2 Données
Même schéma : mélange avec graine, découpage, `[..., None]/255.0` pour **tous** les jeux (même constante 255). Les deux versions d'étiquettes sont conservées :
- `labels_train`, `labels_val`, `etiquettes_test` : **entiers** (pour l'analyse, la matrice de confusion, les comparaisons) ;
- `y_train`, `y_val`, `y_test` : **one-hot** (pour l'entraînement).
`galerie(images, titres, n)` affiche n images sur une ligne avec des titres (`vmin=0, vmax=1` fixe l'échelle de gris pour que les images soient comparables ; `ax.axis("off")` cache les axes).

### 5.3 Le CNN « N »
```python
def cree_N(nom):
    Sequential([Input((28,28,1)),
        Conv2D(16,3,relu,same), MaxPool(2),
        Conv2D(32,3,relu,same), MaxPool(2),
        Flatten(), Dense(64,relu), Dense(10,softmax)], name=nom)
```
**🏫 Formes après chaque pooling et paramètres de la conv 1 :**
- Après conv1 : (28,28,16) → pooling → **(14,14,16)**
- Après conv2 : (14,14,32) → pooling → **(7,7,32)** ; Flatten = 1568
- **Paramètres conv1 = 3·3·1·16 + 16 = 160**.
- Autres couches : conv2 = 3·3·16·32+32 = 4 640 ; dense64 = 1568·64+64 = 100 416 ; dense10 = 64·10+10 = 650. **Total = 105 866.**

`poids_initiaux = [w.copy() for w in N.get_weights()]` stocke les poids initiaux pour que N_aug démarre à l'identique (le `.copy()` est essentiel, sinon on garderait des références vers des tableaux qui changent).
L'exécution prend ≈ 3 min au total ou moins ; **accuracy ≈ 98–99 %** (val). **🏫 « Efficace » ?** Oui : avec ~106k paramètres et quelques époques sur CPU (de quelques dizaines de secondes à quelques minutes), il atteint ≈ 98–99 % : haute accuracy pour un coût modeste. (Note du notebook : pas de Dropout/BatchNorm, donc un modèle simple.)

### 5.4 Matrice de confusion
`cm[i, j]` = nombre d'images dont la **vraie classe est i** et la **classe prédite est j**. **Lignes = vérité, colonnes = prédiction.** Diagonale = bonnes réponses.
```python
def matrice_confusion(vrais, predits):
    matrice = np.zeros((10,10), dtype=int)
    np.add.at(matrice, (vrais, predits), 1)    # pour chaque échantillon k : matrice[vrais[k], predits[k]] += 1
    return matrice
```
`np.add.at` effectue correctement des incréments répétés même quand la même case apparaît plusieurs fois (un simple `matrice[idx] += 1` ne compterait pas les doublons). Mini-exemple : vrais=[5,5,3], predits=[3,5,3] → cm[5,3]=1, cm[5,5]=1, cm[3,3]=1.
- **Normalisée par ligne** : on divise chaque ligne par sa somme → proportions au sein de chaque vraie classe ; la diagonale = **rappel** de chaque classe = VP / (tous les vrais membres de la classe).
- **Précision** de la classe j = cm[j,j] / (somme de la colonne j).
- Accuracy globale = `np.trace(cm) / cm.sum()` (trace = somme de la diagonale).
- `assert cm.sum() == len(x_val)` : chaque image est comptée une fois.
- `erreurs = np.flatnonzero(classes_val != labels_val)` : indices des erreurs (`predict(...).argmax(axis=1)` donne les classes prédites).

**🏫 Case (5,3) ?** Nombre de **vrais 5 prédits 3**. **Classe au plus faible rappel ?** Celle de plus petite valeur diagonale normalisée (exécute-le ; pour MNIST c'est souvent 9, 7, 8 ou 5, vois ta matrice). **Deux confusions orientées les plus fréquentes hors diagonale :** par ex. « vrai 4 → prédit 9 », « vrai 7 → prédit 2/1 », « vrai 5 → 3 », « vrai 9 → 4/7 » (à confirmer avec ta matrice) ; *orientée* veut dire (i→j) ≠ (j→i). **Pourquoi un grand effectif sur la diagonale ne suffit-il pas à comparer les classes ?** Les classes n'ont pas le même nombre d'échantillons (ex. plus de 1 que de 5) ; les effectifs cachent les proportions ; et cela ne dit pas *quelles* classes sont confondues.

Code à compléter :
```python
copie = cm.copy()
np.fill_diagonal(copie, 0)                      # ignorer les bonnes prédictions
top2 = np.argsort(copie, axis=None)[::-1][:2]   # indices aplatis des 2 plus grandes valeurs, décroissant
for k in top2:
    i, j = np.unravel_index(k, copie.shape)     # indice aplati → (ligne, colonne)
    print(f"vrai {i} → prédit {j} : {copie[i, j]} fois")
```
**Au-delà de l'accuracy**, une matrice de confusion révèle *quelles* classes sont confondues, les déséquilibres, les biais systématiques.

### 5.5 Bruit et translations
```python
def bruit_blanc(x, sigma, rng):
    bruit = rng.normal(0, sigma, size=x.shape).astype("float32")   # gaussien N(0, σ²) par pixel, indépendant
    return np.clip(x + bruit, 0, 1)                                # garde l'intervalle valide des pixels
```
**Bruit blanc gaussien** : centré (moyenne 0), indépendant d'un pixel à l'autre, écart-type σ. `clip` ramène les valeurs dans [0,1]. Même graine ⇒ même motif de bruit, seule son amplitude change avec σ.

```python
def translation(x, dx=0, dy=0):
    resultat = np.zeros_like(x)                       # fond noir
    hauteur, largeur = x.shape[1:3]
    if abs(dx) >= largeur or abs(dy) >= hauteur: return resultat   # entièrement sorti du cadre
    xs, xe = max(0, -dx), min(largeur, largeur-dx)    # plage de colonnes source
    ys, ye = max(0, -dy), min(hauteur, hauteur-dy)    # plage de lignes source
    resultat[:, ys+dy:ye+dy, xs+dx:xe+dx, :] = x[:, ys:ye, xs:xe, :]
    return resultat
```
Exemple `dx=5` : xs=0, xe=min(28,23)=23 → colonnes source 0–22 vont aux colonnes 5–27 ; les colonnes 23–27 sortent et sont **perdues** ; les 5 colonnes de gauche deviennent noires. Exemple `dx=−3` : xs=3, xe=28 → source 3–27 → destination 0–24. **dx>0 → droite, dy>0 → bas. Pas de rebouclage** (contrairement à `np.roll` où les pixels réapparaissent de l'autre côté). Tests : `translation(x)` (dx=dy=0) = identité ; un pixel en colonne 27 décalé vers la droite disparaît (`sum()==0`).

```python
SIGMAS = [0.0, 0.15, 0.30, 0.50];  DECALAGES = [0, 2, 4, 6]
def accuracy(modele, x, labels):  predits = modele.predict(...).argmax(axis=1); return np.mean(predits == labels)
```
`np.mean(tableau booléen)` = proportion de True = accuracy. `robustesse` calcule l'accuracy par niveau de bruit (même réalisation de bruit grâce à la même graine) et par décalage (**moyenne des 4 directions** : droite, gauche, bas, haut). `trace_robustesse` trace l'accuracy en fonction de σ et du décalage.

**🏫 Observations (≈) :** propre ≈ 0,99 ; σ=0,15 presque inchangé ; σ=0,30 baisse visible ; σ=0,50 forte baisse. Translations : décalage 2 petite baisse, 4 grosse, 6 sévère.
- *Les convolutions donnent-elles une invariance parfaite à la translation ?* **Non.** Une convolution est **équivariante** par translation (entrée décalée → carte décalée), mais la décision finale passe par **Flatten + Dense** (poids spécifiques à chaque position) et le pooling n'apporte qu'une petite tolérance locale. De plus les chiffres d'entraînement étaient centrés, donc les chiffres décalés sont inhabituels.
- *Déplacement vs perte d'information :* déplacer le chiffre est un problème ; quand le décalage fait sortir des parties du cadre 28×28, des **pixels sont perdus** (recadrage), dommage différent et irréversible.
- *Mêmes effets visuels ?* Non. Le bruit dégrade toute l'image partout (contraste des traits), la translation préserve le contenu mais change **où** il est.

### 5.4b Augmentation de données (N_aug)
Idée : garder N inchangé ; entraîner un **nouveau** réseau `N_aug` (même architecture, **mêmes poids initiaux** via `set_weights(poids_initiaux)`, même nombre d'exemples et de mises à jour) sur un jeu d'entraînement modifié où **environ 1/3 est propre, 1/3 bruité (σ=0,30), 1/3 translaté aléatoirement (|dx|,|dy| ≤ 4)**. Étiquettes inchangées (un 5 décalé ou bruité reste un 5). Découper entraînement/validation/test **avant** d'augmenter ; ne jamais mettre de données de test transformées dans l'entraînement.

```python
groupes = rng.integers(0, 3, size=len(x))           # chaque image reçoit 0, 1 ou 2 au hasard
resultat[groupes==1] ← bruit_blanc(...)             # groupe bruit
resultat[groupes==2] ← translation(...)             # groupe translation ; les images de même décalage (a,b) sont traitées ensemble (vectorisé)
```
Les augmentations sont calculées **une fois** (fixes pendant l'entraînement). Variante : les régénérer à chaque époque (plus de diversité).

**🏫 Résultats attendus (≈) :** propre : perte minime ou inchangé (−0 à −0,5 pt) ; bruit σ=0,30 et décalage 4 : **gros gains** (par ex. +5 à +30 pts). L'amélioration n'est **pas uniforme** : plus forte là où l'entraînement incluait la perturbation ; encore faible pour les perturbations extrêmes (σ=0,5, décalages 6). **Pourquoi garder des images propres ?** Pour que la performance sur la distribution d'origine ne se dégrade pas (le test est surtout propre). **Pourquoi les très grands décalages / bruits très forts restent-ils difficiles ?** L'information est détruite (chiffre partiellement rogné / signal noyé dans le bruit), et ils sortent de la plage vue à l'entraînement.
Remplir le tableau : Propre / Bruit σ=0,30 / Translation 4 (moyenne) avec accuracy N, N_aug, écart.

### 5.4c Protocole de test final
Conditions fixées à l'avance : propre, bruit σ=0,30, décalage de 4 px **vers la droite** (une seule direction : différent de la moyenne sur 4 directions de la validation). Afficher `accuracy(N, x)` et `accuracy(N_aug, x)` pour chacune ; matrices de confusion sur le test propre. Ne pas régler à nouveau après avoir vu ces résultats.

### 5.6 Générer « un 5 » : F = N ∘ G
Chaîne : `z ∈ R^100 → G → x ∈ [0,1]^(28×28×1) → N → p ∈ [0,1]^10`.

- **z (vecteur latent)** : 100 nombres aléatoires tirés d'une loi normale, servant d'entrée au générateur.
- **G (générateur)** : Dense(128, relu) → Dense(256, relu) → Dense(784, sigmoid) → Reshape(28,28,1).
  Paramètres : 100·128+128 = 12 928 ; 128·256+256 = 33 024 ; 256·784+784 = 201 488 ; Reshape 0 → **247 440**.
- **🏫 Pourquoi 784 sigmoïdes + Reshape ?** Les couches Dense sortent des vecteurs plats ; l'image doit être (28,28,1) (entrée de N), donc on produit 784 nombres puis on remet en forme ; la **sigmoïde** garde chaque pixel dans (0,1) comme les vrais pixels normalisés.
- **Dimensions :** G : (100) → (28,28,1). N : (28,28,1) → (10). F : (100) → (10).
- **Données :** des z aléatoires et, pour chacun, la même cible = one-hot de 5 : `[0,0,0,0,0,1,0,0,0,0]`. Perte : **L = −(1/m) Σ log N(G(zᵢ))₅** ; on veut que N mette de la probabilité sur 5.

```python
N.trainable = False                                   # geler AVANT de compiler F
F = keras.Sequential([keras.layers.Input((100,)), G, N], name="F")
F.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
assert len(N.trainable_weights) == 0
assert len(F.trainable_weights) == len(G.trainable_weights)
```
- Geler **avant** `compile` : le statut entraînable est capturé à la compilation.
- **Gelé ≠ gradient bloqué.** La rétropropagation passe toujours *à travers* N (règle en chaîne : ∂L/∂G = ∂L/∂N · ∂N/∂x · ∂x/∂G) mais les poids de N ne sont pas mis à jour ; seuls ceux de G le sont. C'est pourquoi `N.trainable=False` permet quand même à G d'apprendre via F. Ne pas insérer `N.predict`, `argmax` ou des conversions NumPy entre G et N (cela casse la différentiabilité).
- Après `F.fit(z_train, cibles_train, ...)`, le notebook vérifie : variation maximale des poids de N = 0, de ceux de G > 0 (`assert`).
- **🏫 Et si N était entraînable ?** G et N pourraient « tricher » ensemble : N pourrait apprendre à sortir « 5 » quoi que G produise (par ex. en ignorant l'entrée), détruisant son rôle de classifieur de chiffres ; l'objectif devient trivial et sans sens.

L'accuracy de `F` = proportion d'images générées classées 5 par N (pas l'accuracy MNIST, pas la qualité visuelle). Évaluation sur de nouveaux vecteurs latents : `z_inedits` (1000 nouveaux z). Affiche la proportion classée 5 (≈ 1,0), le score softmax moyen pour 5 (≈ 0,99), l'écart-type moyen des pixels entre images (indicateur grossier de diversité).

**🏫 Les images ressemblent-elles à des 5 ?** Typiquement **non, ou à peine** : des taches/textures bruitées que N déclare 5 avec confiance, comme des **exemples adverses** : N a seulement appris à séparer de vrais chiffres, pas à rejeter des entrées absurdes. **Un softmax de 0,99 garantit-il la ressemblance ?** Non : la confiance d'un classifieur n'a de sens que pour des entrées semblables à son entraînement. **G utilise-t-il z ?** Souvent faiblement ; les images se ressemblent presque toutes (faible écart-type des pixels) → **effondrement de modes (mode collapse)** : la perte demande seulement « classé 5 » ; une seule image qui marche pour tous les z est optimale ; rien ne récompense la diversité ou le réalisme.

Vérifications supplémentaires : `N_aug` classe-t-il les images de G comme 5 ? (Souvent moins que les 100 % de N : les images sont ajustées pour tromper N spécifiquement ; ça ne se transfère que partiellement ; ce n'est pas non plus une preuve de réalisme.) **Interpolation :** `z = (1−α)z₀ + α z₁` pour α dans [0,1] ; si G est pertinent, les images changent en douceur ; en cas d'effondrement, elles bougent à peine.

**Ce n'est pas un GAN :** dans un GAN, un **discriminateur** est entraîné (en alternance avec G) à distinguer images réelles et générées, ce qui pousse G vers la distribution des vraies données. Ici N est un classifieur de chiffres gelé. Pour apprendre de vrais 5 réalistes, il faudrait de vraies images de 5 dans l'objectif (discriminateur, VAE, etc.).

### 5.7 Réponses de synthèse (5 points)
1. **Matrice de confusion > accuracy :** montre *quelles* classes sont confondues, le sens (i→j), rappel/précision par classe, les déséquilibres.
2. **CNN bon sur MNIST propre mais fragile :** il est entraîné sur des chiffres propres et centrés ; les perturbations sortent les entrées de la distribution d'entraînement ; équivariance ≠ invariance ; Flatten+Dense et pooling ne sont pas invariants ; le bruit change les statistiques des pixels.
3. **Augmentation :** les images transformées gardent l'étiquette d'origine (un 5 bruité/décalé reste un 5), ce qui fait voir au modèle des entrées plus variées sans nouvelles étiquettes. Limites : ne couvre que les perturbations auxquelles on a pensé ; peut baisser l'accuracy sur le propre ; des transformations extrêmes détruisent l'information ou changent l'étiquette ; une augmentation fixe est moins diverse qu'une augmentation par époque ; le jeu de test doit rester intact.
4. **`N.trainable=False` permet quand même d'entraîner G :** les poids sont gelés mais les gradients traversent toujours N jusqu'aux sorties de G, et l'optimiseur ne met à jour que les poids entraînables (ceux de G).
5. **« Être classé 5 » ≠ réalisme/diversité :** la perte ne mesure que la probabilité de sortie de N ; beaucoup d'images irréalistes (et une seule image) la satisfont.

---

## 6. Pièges & bugs (check-list anti-erreurs)
1. S1 : `x_test / y_test.max()` (des étiquettes !) → mauvaise mise à l'échelle ; il faut `/255`.
2. S1 : coquille `hystory` / `plt` seul (pas de `.show()`).
3. S2 : premier print répété (train deux fois).
4. S2 : nommage softmax vs sigmoid ; `"leaky relu"` vs `"leaky_relu"` → KeyError.
5. S2 : `validation_split` prend les **20 % derniers** (S1), alors que S2 utilise `validation_data`.
6. S3 : `nom_choisi =  # ...` est une SyntaxError tant qu'il n'est pas rempli.
7. S3 : avec Dropout, les métriques train & val sont mesurées dans des conditions différentes ; avec L2, la loss inclut la pénalité.
8. S4 : le score de translation en validation = moyenne de 4 directions ; le test utilise la droite seulement.
9. S4 : `.copy()` sur les poids initiaux ; geler N **avant** de compiler F.
10. Partout : choisir avec la validation ; ne toucher au test qu'une fois.

---

## 7. Aide-mémoire d'une page

**Formules**
- Params Dense = `n_entrée·n_sortie + n_sortie`
- Params Conv2D = `(k_h·k_w·C_entrée + 1)·C_sortie`
- Sortie de conv (valid) = `n − k + 1` ; same → `n` (pas de 1)
- MaxPool(2) → divise H et W par 2, 0 paramètre
- Mises à jour par époque = `⌈N/lot⌉`
- sigmoïde = `1/(1+e^-x)` ; relu = `max(0,x)` ; softmax = `eˣⁱ/Σeˣʲ`
- Entropie croisée (one-hot) = `−log p_vraie`
- Accuracy = bonnes / total ; rappel_c = cm[c,c]/somme ligne ; précision_c = cm[c,c]/somme colonne
- Pénalité L2 = `λΣw²` ; Dropout de taux r : met à zéro une fraction r, seulement à l'entraînement

**Nombres de paramètres à retenir :** S1/S2 dense de référence 101 770 (16 → 12 730 ; 512 → 407 050) ; S3 dense 1 796 010 vs CNN 26 698 ; S4 N 105 866, G 247 440.

**Extraits de code à connaître par cœur**
```python
x = x.astype("float32")[..., None] / 255.0
y = keras.utils.to_categorical(labels, 10)
model.compile(optimizer=keras.optimizers.Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
h = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=8, batch_size=128)
keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
model.evaluate(x_test, y_test, return_dict=True)
model.predict(x).argmax(axis=1)
```

**Vocabulaire FR ↔ EN :** époque = epoch ; lot = batch ; poids = weights ; biais = bias ; perte = loss ; taux d'apprentissage = learning rate ; surapprentissage = overfitting ; arrêt anticipé = early stopping ; rappel = recall ; matrice de confusion = confusion matrix ; augmentation de données = data augmentation ; partage des poids = weight sharing ; carte de caractéristiques = feature map.

---

## 8. Auto-test (cache les réponses !)

1. Forme de `x_train` pour MNIST ? → (60000,28,28). Après `[..., None]` ? → (60000,28,28,1).
2. Que renvoie `to_categorical(7,10)` ? → vecteur de longueur 10 avec un 1 à l'indice 7.
3. Pourquoi `categorical_crossentropy` et pas l'accuracy comme perte ? → dérivable, pénalise la confiance sur les mauvaises réponses.
4. Paramètres de Dense(128) après Flatten de 28×28 ? → 100 480.
5. À quoi sert la validation ? → comparer/choisir hyperparamètres et époques. Le test ? → une estimation finale non biaisée, une seule fois.
6. Lots de 64 avec 500 images : mises à jour par époque ? → 8.
7. Signature du surapprentissage ? → loss train ↓, loss val ↑ de façon persistante.
8. Pourquoi la loss val peut monter avec une accuracy stable ? → confiance excessive croissante sur les erreurs.
9. Dropout à l'inférence ? → inactif.
10. `patience=3` ? → arrêt après 3 époques sans amélioration. `restore_best_weights` ? → retour à la meilleure époque.
11. Sortie de `padding="valid"` pour 28×28 et filtre 3×3 ? → 26×26. `same` ? → 28×28.
12. Pourquoi un CNN a moins de paramètres ? → poids partagés, connexions locales.
13. Paramètres de Conv2D(16,3) sur 8 canaux ? → 3·3·8·16+16 = 1168.
14. Un CNN est-il invariant par translation ? → équivariant seulement ; le classifieur complet n'est pas invariant.
15. Sens de cm[5,3] ? → vrai 5 prédit 3.
16. Rappel vs précision depuis une matrice de confusion ? → diag/somme ligne vs diag/somme colonne.
17. Pourquoi garder des images propres dans l'augmentation ? → conserver la performance sur le propre, ne pas décaler entièrement la distribution.
18. Pourquoi `N.trainable=False` laisse G apprendre ? → les gradients traversent N, seul G est mis à jour.
19. Pourquoi G peut produire du n'importe quoi que N appelle « 5 » ? → N est juste un classifieur, l'objectif ignore réalisme/diversité ; type exemples adverses.
20. Pourquoi ce n'est pas un GAN ? → pas de discriminateur entraîné en alternance contre de vraies images.
21. Pourquoi mélanger les meilleurs hyperparamètres individuels peut échouer ? → interactions.
22. Quelle ligne contenait le bug de normalisation par les étiquettes ? → `x_test.astype(...) / y_test.max()` en S1.
23. Ordre des axes dans `x[17,12,4]` ? → image, ligne, colonne.
24. Effet d'un taux d'apprentissage énorme ? → dépassement, instabilité/divergence ; minuscule → apprentissage très lent.
25. Un lot plus grand signifie-t-il moins de mises à jour par époque ? → oui, et des gradients moins bruités, généralement sous-entraîné à nombre d'époques fixé.

Bon courage pour demain ! Si tu veux, envoie-moi tes propres chiffres issus des exécutions et je t'aide à rédiger les conclusions des tableaux.
