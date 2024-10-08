# Team Train to Failure - [BirdCLEF 2024 @ Kaggle](https://www.kaggle.com/competitions/birdclef-2024)

## Developing Team - Train to Failure

### Bódi Vencel (VBW5N9) - Villamosmérnöki és Informatikai Kar (Faculty of Electrical Engineering and Informatics)
Damien Karras vagyok


### Mitrenga Márk (OLLNTB) - Közlekedés és Járműmérnöki Kar (Faculty of Transportation Engineering and Vehicle Engineering)
Én meg a sátán


## BirdCLEF competition

The **Kaggle BirdCLEF 2024** competition aims to challenge participants to develop machine learning solutions for recognizing bird species through audio-based identification, focusing particularly on understudied Indian bird species in the Western Ghats, a biodiversity hotspot. The task relies on Passive Acoustic Monitoring (PAM) to support large-scale bird observation and enhance the accuracy of biodiversity assessments. Participants will need to process continuous audio recordings to identify bird species, with a special emphasis on nocturnal and endangered species that have limited training data available.

The competition's broader goal is to support conservation efforts by providing technological solutions that help monitor bird populations and guide effective conservation actions. By advancing such innovations, the competition aims to improve the accuracy and regularity of avian population surveys, thereby aiding in biodiversity protection efforts.



# Tervezet, lépések
Milyen lépésekkel jutunk el a kész házihoz?
## Koncepcióalkotás
### Elméleti anyagok:
 - [Audio + CNN](https://www.analyticsvidhya.com/blog/2021/06/how-to-detect-covid19-cough-from-mel-spectrogram-using-convolutional-neural-network/)

Milyen módszerrel akarunk dolgozni? RNN? CNN-t eresztünk rá a spektogramokra? Milyen modellel szeretnénk dolgozni? Mi írjuk a modellt, vagy keresünk egy előre megírtat? CNN + RNN? RNN esetén LSTM vagy GRU?
## 2 Adathalmaz basztatása
### Elméleti anyagok:
- [Mel spektogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)

Ki kiell nyerni az adatokat az adathalmazból. Meg kell ismerni a struktúrát. Mel spektogramok elkészítése -> Audioból képanyag gyártása

### Adatok az adatokról (train_metadata.csv alapján)
 - 182 class
 - 24459 hangfájl a train_audioban
 - Fontosabb labelek:
    - primary_label: A class neve, lehetne One - Hot kódolni
    - secondary_label: Ha másik madár is van a felvételen, hasznos lehet, One-Hottal faszán bele lehet tenni
    - type: Erősen eltér fajon belül is, hogy mi a fütyfürüttyöt csinál a madár, és annak milyen hangja van. Lehetne ezt is One-hot kódolni
    - latitude, longitude: elvileg ezek a repkedő cuccok is dialektusokban fütyfürüttyölnek, tehát kellhet
    - scientific_name, common_name: Ha menők akarunk lenni, akkor csiálunk belőle interpretable labelt
    - filename: Ha jól látom a fájlok nem sorban vannak, kellhet
 - unlabeled_soundscapes --> a végén majd a confidence-t tudjuk majd megfigyelni





## Acknowledgements

The project was developed within the framework of the subject "Deep Learning in Practice with Python and LUA" (BMEVITMAV45) at the Faculty of Electrical Engineering and Informatics of Budapest University of Technology and Economics.

We used generative airtificial intelligence for the following purpuses:
 - code refactoring (**not writing code!**)
 - code commenting