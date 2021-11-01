# MAP511 - EA première période

Le calcul par méthode de monte-carlo permet d'estimer avec précision des intégrales ou des espérences parfois très complexes à calculer analytiquement. Cependant, la simulation des échantillons peut être très coûteuse et ainsi rendre les calculs de monte-carlo prohibitivement lent. En particulier, dans le cas d'un portfolio comportant un grand nombre d'obligations, calculer un échantillon revient à tirer autant de variables aléatoires qu'il y a d'obligations dans le portfolio, ce qui peut être très coûteux. On peut donc s'intéresser à l'approximation d'un tes portfolio par un méta-model statistique qui capture la distribution des risques du portfolio en restant plus simple à simuler.

Nous nous intéressons ici à l'approximation des fonctions de seuil en utilisant les polynômes du cahos. Dans un premier temps, nous nous intéressons aux propriété de convergence de ces estimateurs. Puis nous construirons un méta-modèle pour simuler de manière efficace un portfolio d'obligations.

# Polynômes du cahos

## Rappels sur les polynômes du cahos

On considère une distribution $D$, de fonction de densité $\mu$, de moments finis à tout ordre. Notre problème est d'étudier le comportement de $Y(X)$ où $X\simD$.

On défini le produit scalaire dans $L^2$ : $<f,g> = \int_{-\infty}^{+\infty} f(x)g(x)\mu(dx)$. L'idée est de décomposer Y dans une base de polynômes orthonormés pour le porduit scalaire ainsi défini. On appelle les polynômes de cette base les polynômes du cahos associés à la distribution $\mu$. 

Calculer cette base de polynôme implique en générale de calculer des intégrale possiblement complexe. Heureusement, dans le cas de distributions standards, les polynômes du cahos sont connus. Ainsi, on a les correspondances suivantes : 

- loi normale (0,1) -> polynômes de Hermite
- loi gamma ($\alpha$+1, 1) -> polynômes de Laguerre
- loi 1-2*Beta($\alpha$+1, $\beta$+1) -> polynômes de Jacobi
- loi uniforme (-1, 1) -> polynômes de Legendre

Ces familles de polynômes forment ce que l'on appelle les Classical orthogonal polynomials set (COPS). 

## La fonction seuil

On cherche maintenant à décomposer la fonction $Y$ sur la base des polynômes du cahos pour une distribution donnée. Encore une fois, pour calculer les coefficients de cette décomposition, il faut en général calculer une intégrale complexe. Heureusement, dans le cas de certaines fonctions simples, les valeurs de ces coefficients peuvent être calculés dirrectement.

En particulier, [cet article](https://hal.archives-ouvertes.fr/hal-03199734/document) propose deux moyens de calculer ces coefficients dans le cas de la fonction de seuil $Y(x) = 1_{c<x}$. Une première méthode utilise une relation de récurrence entre les différents coefficients et la deuxième méthode exprime dirrectement les coefficients en utilisant les fonctions Gamma et Beta ainsi que l'évaluation des polynômes du cahos en certains points.

Le calcul de ces coefficients en utilisant chacune des deux méthodes a été implémenté dans le fichier python/polynomial.py.

## Vitesse de convergence

Pour une distribution donnée, on peut donc exprimer la fonction seuil comme une série des polynômes du cahos. Mais pour être réellement utile, cette on ne peut utiliser qu'une partie de cette série : il faut procéder à une troncature. La question est alors de savoir s'il est possible d'estimer l'erreur effectuée au moment de la troncature à l'ordre $N$. L'article précédement cité montre que l'on peut effectivement estimer ces vitesse de convergence à $N^\frac{-1}{2}$ quand N tend vers $+\infty$. On observe expérimentalement cette vitesse de convergence pour tous les COPS. 

On remarque aussi que les bornes résentées dans le papier sont "tight". Elles sont essentiellement optimales pour toutes les valeurs de seuil et tous les ordres de troncature. Pour mettre en évidence ce phénomène, on calcule le ratio entre l'erreur $L^2$ effective et la prévision de l'erreur. On remarque que les ratio obtenus sont très stables pour toutes les valeurs de N et toutes les valeurs de seuil. 

Enfin, le papier remque qu'en effectuant une transformation de la variable aléatoire, si on connais les fonctions de distribution cumulatives et leurs inverses, on peut passer d'une loi à une autre. Au cours de cette transformation, le seuil est modifié mais la probabilité $P(X<c)$ où c est le seuil reste inchangée. On peut alors se demander, à quantile fixé, quel est la loi à utiliser pour obtenir la plus petite erreur de troncature. Il est montré dans ce papier que ce résulata dépend du quantile d'intérêt. Ainsi, les polynômes de chebychev (jacobi avec $\alpha = \beta = \frac{-1}{2}$) offrent la plus petite erreur pour des quantils pas trop extrèmes. Cependant, pour des quantils extrèmes (prochent de 0 ou proche de 1), un autre choix des paramètres de polynômes de jacobi donnent des erreurs plu petites. 

L'implémentation de ces visualisation expérimentale est effectuées dans les fonctions du fichier python/L2_error.py.

# Portfolio d'obligations

Un sujet d'intérêt pour les banques et autres structures financières est la distribution de leur gain (ou perte) à horizon donné. On s'intéresse ici à une structure disposant d'un portfolio composé d'un grand nombre d'obligations. Une obligation est un produit financier qui oblige le détenteur à payer une certaine somme (fixée) sous certaines conditions (évolution du marché). 

Plus précisément, on s'intéresse à la distribution de $L = \sum_{k=1}^K l_k Y_k$ où K est grand. Les l_k sont les prix des obligations et les Y_k sont des variables a de bernouilly indiquant si l'obligation est due ou non. On suppose ici une dépendance de copule gaussienne entre les Y_k. En particulier, on a $Y_k = 1_{c_k<X_k}$ où $c_k$ est un seuil et $X_k = \rho_k Z + \sqrt{1-\rho_k^2} \varepsilon_k$. $Z \sim N(0,1)$ simule l'état du marché global, l'ensemble des $\varepsilon_k \sim N(0,1)$ correspondentaux variabilités individuelles de chaque obligation et les $\rho_k$ sont des facteurs de correlation entre -1 et 1.





