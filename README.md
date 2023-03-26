# CharlesHendrix
## Autore
Carmine Iemmino
## Obiettivo del progetto
L'obiettivo del progetto è sviluppare un sistema software che, dato in input numero di battute e scala da usare, permetta di comporre musica in maniera automatica. In questo caso la scelta per la risoluzione del problema è virata sugli algoritmi genetici.
## Risorse della repository 
**src**: contiene il codice sorgente del sistema <br />
**xmls**: contiene files .xml da cui è stata isolata la melodia principale. In una precedente versione del sistema era possibile utilizzarli per effettuare mash-ups di composizioni. Allo stato attuale ciò non è possibile ma, tramite piccole modifiche al codice che verranno effettuate in seguito, la funzionalità tornerà disponibile. <br />
**documentation**: contiene la documentazione relativa al sistema
## Utilizzo del sistema
Il file main del sistema è *Runner.py* che permette di creare una composizione musicale. Tramite codice si possono effettuare modifiche ai parametri del *Composer*: numero di battute, durata dell’elemento musicale più piccolo, scala (istanza di *Key*), una progressione di accordi, una lista di intervalli considerati spiacevoli, i pesi dei criteri della funzione di fitness, le probabilità di mutazione dei diversi elementi musicali e un valore booleanoche indica se comporre utilizzando tutte le note o esclusivamente quelle della scala oppure ai parametri tipici di un algoritmo genetico, come numero di generazioni, numero di individui per generazione, numero di genitori che si accoppiano, etc. <br />
A riga 24 di *Runner.py* è presente un'istruzione che permette di generare una progressione di accordi tramite algoritmo genetico interattivo implementato in *Chords.py*. Quest'istruzione è di default commentata, permettendo eventualmente di definire in maniera manuale una progressione di accordi tramite l'istanziazione di un *Composer*. <br />
Per l'esecuzione è inoltre fondamentale aver installato la libreria *pygad*: https://pygad.readthedocs.io/en/latest/index.html e la libreria *music21*: http://web.mit.edu/music21/doc/index.html <br />
Per riprodurre la composizione generata è inoltre necessario avere un software di manipolazione musicale che permetta la lettura di file *.midi* o *.musicxml*. Questo programma va poi associato con *music21*, in modo che quando venga chiamato il metodo *show()* su un oggetto *Stream*, questo venga riprodotto sul software musicale.
