# [Дефектоскопия на основе компьютерного зрения](https://aviahack.mai.ru/tasks/4)

Авиахакатон 2021

## Краткое описание
Поиск и классификация повреждений на элементах авиадвигателей при производстве для повышения безопасности пассажирских перевозок.

## Текущая ситуация
При производстве лопаток авиационных газотурбинных двигателей необходимо обеспечить высокое качество заготовок, получаемых методом литья по выплавляемым моделям. Перед проведением механической обработки заготовки подлежат визуальному контролю, направленному на выявление дефектов литья (раковины, трещины, пробой, засор, дефекты выходной кромки, сколы, плохое клеймение).

## Проблема
Визуальный контроль требует большой внимательности и является рутинным ввиду большого количества контролируемых объектов, что при выполнении данной операции человеком может привести к пропуску дефекта. Если дефект не будет обнаружен на этапе визуального контроля, бракованная заготовка будет направлена на механическую обработку, что приводит к дополнительным издержкам в процессе производства. Подготовка квалифицированного специалиста для выполнения визуального контроля сложных геометрических объектов требует длительного времени и специальной аттестации (это приводит к высокой стоимости специалистов для предприятия). Требуется использование дополнительных механизмов, позволяющих с необходимой степенью достоверности выявлять и классифицировать дефекты при помощи машинного зрения в качестве интеллектуального помощника контролёру. Учитывая, что лопатки имеют различные габариты, а также наличие перечня дефектов, частота, которых не велика, решать задачу посредством традиционных решений по машинному зрению на основе классических алгоритмов, основанных на фиксированных правилах затруднительно.

## Задача
Требуется разработать программное обеспечение для машинного обучения на основе анализа изображений, которое позволит производить:
1. Загрузку объекта исследования (обучающий набор)
2. Разметку и классификацию дефектов вручную/автоматически
3. Обучение одной/нескольких моделей машинного обучения и выявление лучшей модели
4. Производить загрузку фотографий и выявление, идентификацию дефектов на основе загруженного изображения
5. Формировать отчет об обнаруженных дефектах с их указанием на загруженном изображении.
6. Обнаружение дефектов на загруженном снимке
7. Обеспечить доп8. олнение обучающего набора новыми дефектами
8. Дообучение обученной модели машинного обучения
По результатам обработки изображения программа должна формировать отчет о годности заготовки и отсутствии / наличии дефектов с их указанием на загруженном изображении.

## Данные
1. Training Dataset* (120 изображений заготовок) c разметкой и идентификацией различных видов дефектов: раковины; трещины;
2. Additional Dataset* (20 изображений заготовок), иных дефектов не представленные в Training Dataset для дообучения модели.
3. Validation Dataset* (5 изображений) заготовок с дефектами из Training Dataset и Additional Dataset* для выявления и идентификации дефектов (участники получат после 3 чек-поинта)

## Постановщик
https://www.uecrus.com/rus/