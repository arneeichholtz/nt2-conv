Ik heb drie context variables toegevoegd aan config.yml:
1. store_speech: wil je de input spraak bewaren voor latere analyse of training.
                 Dit is alleen geimplementeerd voor fasterWhisper en de module asrLou.py.
2. talking_head: pytoon of MakeItTalk. 
                 Alleen de gekozen talkingHead module wordt geimporteerd. Op die manier
				 voorkom ik naamgevingsconflicten. 
3. tts_engine:   gtts of piper
                 Het gezicht in de huidige versie van MakeItTalk past niet bij de 
				 mannenstemmen van PIPER. Ik ga kijken of er een acceptabel mannenhoofd
				 te vinden is. 
				 
Het script main_talkingHead.py handelt de keuzes af.
Het is nodig om het statement op regel 37 aan te passen. De expliciete chdir() is 
waarschijnlijk in de meeste gevallen overbodig, maar als het script gestart wordt 
vanuit een andere working directory gaat het zeker fout omdat subdirectories met
essentiele informatie niet gevonden kunnen worden. 

asrLou.py zit een klein beetje anders in elkaar dan jouw asr.py. Het belangrijkste verschil 
is dat tqdm nu gebruikt wordt als indicator dat er gesproken kan worden. Waarschijnlijk
is tqdm al geinstalleerd, maar helemaal zeker weet ik dat niet. Zo nodig moet je
pip install tqdm doen. 
Bovendien bevat asrLou de optie om de input spraak te bewaren. 
Ik maak op dit moment geen gebruik van de mogelijkheid om fasterWhisper informatie te geven 
over de voorafgaande prompt(s) en om een lijst van 'hotwords' mee te geven. Dat kan de 
herkenperformance wel verbeteren. Het meegeven van prompts is niet lastig om te implementeren;
een mogelijke lijst van hotwords is sterk afhankelijk van de context die lastig te 
voorspellen is, maar als je die lijstjes zou hebben is de implementatie makkelijk.

pytoon is het eerste talking head. Ik heb op grote schaal veranderingen aangebracht in 
de code die ik kreeg toen ik het pakket ophaalde van https://github.com/lukerbs/pytoon. 
Als de directory die ik gestuurd heb je working directory is, zou je dit talking headmoeten 
kunnen laten praten door python pytoon.py te zeggen. 
Het enige pakket dat misschien ontbreekt is de beeldverwerking in cv2. Als je dat pakket
nog niet hebt, kun je het installeren met pip install opencv-python. 
pytoon haalt informatie op uit de subdirectory assets. 

MakeItTalk is het andere talking head. Ik heb het ooit opgehaald uit 
https://github.com/yzhou359/MakeItTalk
maar in het script dat ik je stuur is heel veel aangepast; vooral om te proberen alles
een beetje sneller te maken. (En om allerlei versie-conflicten op te lossen.) 
MakeItTalk heeft de subdirectories src, thirdparty en utils nodig waar additionele code 
uit gehaald wordt. En de subdirectory examples, waar tegen de naam van die subdirectory ook
een paar sub-subdirectories in staan waar getrainde modellen in opgeborgen zijn. 
Die modellen zijn groot, zo groot dat ik ze niet via email kan/wil sturen. 
Als het goed is, zou je geen additionele pakketten hoeven te installeren. 


