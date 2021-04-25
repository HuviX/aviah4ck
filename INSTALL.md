# Установка

## Локальная разработка
```
make init
make run
```

Создать базу с дамми данными
```
PYTHONPATH=. python app/db/cli.py
```

## Для деплоя

```
make up
```