# Проект - разработка веб-сервера для семантического анализа новостей о различных акциях и компаниях

## Цель - разработка приложения, которое по тексту новости и заголовку смогла производить семантическую оценку, и в качестве результата указывать является ли новость положительной или отрицательной

## Стек  

    - Python 3.12+

    - pandas

    - tensorflow (tf-keras)

    - scikit-learn

    - ai-model - (CNN (+ XGBoost в будущем) или BERT (RuBert))

    - Для REST-API используется FAST-API

## План разработки

    - Инициализация репозитория и заполнение readme.md

    - Создание модуля для аналитики и подготовки датасета с данными для обучения модели
        - Подключить pandas
        - Загрузить датасет в проект
        - Проанализировать датасет и подтоготовить его к загрузке в токенизатор и дальнейшему обучению модели на нем
        - Создание, подготовка и  сохранение токенизатора
        - Создание и обучение модели
        - Проверка качества работы модели на тестовых данных, дополнительная настройка и обучение модели, при необходимости
        - Сохранение модели

    - Создание веб-сервера на Fast-API и создание энд-поинта для анализа сообщения и возвращения результата работы модели
        - Создать базовое FastAPI приложение, проверка настроек и работоспособности
        - Импорт сохраненного токенизатора и модели
        - Создание энд-поинта, который получает текст новости, токенизирует этот текст, загружает его в модель, и отдает результат работы

    - Тестирование созданного сервиса и модели

### Опциональные задачи  

    - Создание докер-контейнера, в котором запускается Fast-API сервер
    - Создание второй модели (чтобы существовали и BERT и CNN, для более точного предсказания семантики новости, и анализа какая из моделей лучше покажет себя в работе)

## Для запуска приложения необходимо

    - Скопировать данное приложение из GIT на свою личную машину 

`git clone https://github.com/TerentiiDHTB/End-of-Semester-Examination---Artificial-Intelligence.git`  

    - Опционально: поднять у себя виртуальное окружение если у вас MacOS, то вместо python надо указать python3, и при активации указывать не только путь до виртуального окружения, но и команду source

`python -m venv venv`

`.venv/Source/activate`

    - Установить все необходимые пакеты
`pip install -r requirements.txt`  

    - Запустить сервер FastApi
`fastapi dev main.py`

    - Работать с сервером по адресу, указанному после его запуска

    - При необходимости можно переобучить модель. Для этого необходимо запустить файл ai-model.py, в директории ai-model (например python ai-model/ai-model.py). После запуска и обучения модель и токенизатор автоматически сохранится в той же директории.
