# MAP511 - EA première période

Le calcul par méthode de monte-carlo permet d'estimer avec précision des intégrales ou des espérences parfois très complexes à calculer analytiquement. Cependant, la simulation des échantillons peut être très coûteuse et ainsi rendre les calculs de monte-carlo prohibitivement lent. En particulier, dans le cas d'un portfolio comportant un grand nombre d'obligations, calculer un échantillon revient à tirer autant de variables aléatoires qu'il y a d'obligations dans le portfolio, ce qui peut être très coûteux. On peut donc s'intéresser à l'approximation d'un tes portfolio par un méta-model statistique qui capture la distribution des risques du portfolio en restant plus simple à simuler.

Nous nous intéressons ici à l'approximation des fonctions de seuil en utilisant les polynômes du cahos. Dans un premier temps, nous nous intéressons aux propriété de convergence de ces estimateurs. Puis nous construirons un méta-modèle pour simuler de manière efficace un portfolio d'obligations.

# Polynômes du cahos

On considère une distribution $D$, de fonction de densité $\mu$, de moments finis à tout ordre. Notre problème est d'étudier le comportement de $Y(X)$ où $X\simD$.

On défini le produit scalaire dans $L^2$ : $<f,g> = \int_{-\infty}^{+\infty} f(x)g(x)\mu(dx)$. L'idée est de décomposer Y dans une base de polynômes orthonormés pour le porduit scalaire ainsi défini. On appelle les polynômes de cette base des polynômes du cahos. 
