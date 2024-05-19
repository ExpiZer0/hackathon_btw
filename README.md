# hackathon_btw
Для начала работы приложения нужно выполнить несколько следующих шагов.
- Клонирование репозитория
```
git clone https://github.com/ExpiZer0/hackathon_btw.git
```
- Устанавливаем venv
```
cd hackathon_btw
py -m venv venv
```
- Устанавливаем django
```
venv\Scripts\activate
pip install django
```
В принципе, с этим уже можно работать.
Только нужно помнить что для сборки static файлов следует
после каждого изменения js/css/img файлов.
Для этого надо в папке с manage.py запустить команду
```
py manage.py collectstatic
```
[Text: комаров
Probabilities: [0.91189545, 0.9836024, 0.7703604, 0.6825426, 0.60875124, 5.7843572e-05, 4.7298736e-06]](/samples/sample_0.jpg)

[Text: авуста
Probabilities: [0.9959135, 0.9756692, 0.20369764, 0.001068311, 1.0913084e-05, 3.3146532e-06]](/samples/sample_1.jpg)

[Text: 190
Probabilities: [0.9746309, 0.38968942, 0.118148714]](/samples/sample_3.jpg)
