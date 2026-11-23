# Telefonla çekilen test fotoğrafları

Fotoğrafları bu klasöre **aşağıdaki adlarla** koyun. Beklenen sonuçlar
`predict_test_photos.py` içindeki `EXPECTED` sözlüğünde tanımlıdır.

| Dosya | İçerik | Beklenen |
|---|---|---|
| sise.jpg | Mutfak tezgâhında şeffaf plastik su şişesi | bottle |
| kasik_catal.jpg | Granit tezgâhta metal kaşık ve çatal | spoon, fork |
| meyveler.jpg | İki portakal ve bir kivi | orange ×2 (kivi COCO'da yok) |
| kitaplar.jpg | Masada üst üste kitaplar ve defterler | book (birden fazla) |
| bardak_kupa.jpg | Cam bardak ve seramik kupa | cup ×2 (cam bardak wine glass ile karışabilir) |
| sandalye.jpg | Kahverengi deri, krom ayaklı sandalye | chair |
| koltuk.jpg | Gri kumaş kanepe, üzerinde yeşil yastıklar | couch |

Not: iPhone fotoğrafları `.heic` formatındaysa önce `.jpg` formatına dönüştürün.
