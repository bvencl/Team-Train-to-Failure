# tervezet, lépések
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
