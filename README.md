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
/samples/sample_0.jpg
