#Clustering: NBA Players

##Data

De data is afkomstig van sports-reference.com.
Het gaat om data per speler per 100 bal bezittingen.

<https://www.sports-reference.com/blog/2014/06/introducing-per-100-possessions-statistics/>

De dataset heeft een hoog aantal kolommen (dimensies).

Het is je misschien opgevallen dat in deze versie van de dataset de labels (= posities op het veld) aanwezig zijn. We gaan deze niet gebruiken tijdens onze oefening. Enkel aan het einde om te controleren.

##Doel

We gaan op zoek naar verschillende spelers type en dit mbv unsupervised learning: clustering.

Het doel is een clustering algoritme toe te passen op deze dataset en daarna alle spelers te plotten op een grafiek.

##Setup

In deze oefening worden enkele libraries gebruikt. Als je deze nog niet hebt geinstalleerd doe dit dan mbv pip

```
pip install numpy pandas sklearn mpl_toolkits matplotlib
```

indien je verschillende versies van python op je machine hebt staan moet je misschien ```pip3 install``` gebruiken

##Stappen

1. Lees de data in via pd (<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>)

2.  Clean de data
	 Er wordt gebruik gemaakt van de 'drop' functie in pd om enkele kolommen te verwijderen. Zorg ervoor dat de kolom 'Pos' wordt bewaard in variabele (bv. y) en verwijder daarna ook volgende kolommen
	 ```
	 ['Player', 'Pos', 'G', 'Player_ID','url', 'Status'
	 ```
	 Bewaar de resterende data in een variabele X
	 
3. Gebruik de [Standard Scalar](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) voorzien in SciKit om de waarden te normaliseren.

4. Reduceer de dimensie van X mbv [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?highlight=pca#sklearn.decomposition.PCA). Probeer de oefening eerst af te werken met 2 dimensies, later eventueel met 3.

5. Pas nu het [Kmeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?highlight=k%20means#sklearn.cluster.KMeans) algoritme toe. Experimenteer met het aantal clusters.

Plot deze op de matplot grafiek. Probeer de bijhorende y waarden dmv kleur toe te voegen aan de grafiek.


## Resources

<https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html>

Bekijk ook zeker de pdf met daarin het deel over PCA en Kmeans.








