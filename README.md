# AI Shorts Cutter

Утилита для нарезки длинных видео в короткие вертикальные клипы (Shorts/Reels/TikTok) с автосегментацией, субтитрами и оверлеями.

Поддерживаются два режима анализа:

- `speech` - сегментация по речи (паузы, транскрибация, динамические субтитры).
- `beat` - сегментация по музыкальному ритму.

Проект включает:

- CLI (`python -m final_project.main`)
- GUI на `tkinter` (`python -m final_project.gui`)

## Возможности

- Нарезка одного или нескольких входных видео.
- Автоматический выбор кандидатов клипов с ограничением по длительности.
- Вертикальный формат `1080x1920` с адаптацией кадра.
- Оверлеи:
- размытие фона (`blur`)
- стикер/GIF/видео-стикер
- дополнительная аудиодорожка
- Субтитры:
- включение/выключение
- фон под текстом
- авто-подбор размера
- выбор системного шрифта или файла шрифта
- Параллельный рендер нескольких клипов.
- Использование NVENC при доступности (fallback на `libx264`).

## Структура проекта

- `src/final_project/main.py` - CLI и сборка `ProcessingRequest`
- `src/final_project/gui.py` - desktop GUI
- `src/final_project/generator.py` - основной пайплайн обработки
- `src/final_project/segmentation.py` - анализ речи/ритма и сегментация
- `src/final_project/models.py` - модели запросов и стили субтитров
- `src/final_project/core/` - базовые редакторы и интерфейсы
- `src/final_project/decorators/` - blur/sticker/sound/subtitles
- `tests/` - unit-тесты

## Требования

- Python `3.11+`
- FFmpeg в `PATH`

Для ускорения рендера (рекомендуется):

- NVIDIA GPU с поддержкой NVENC
- CUDA-совместимая сборка PyTorch

## Установка (Windows / PowerShell)

```powershell
cd C:\Users\S1NTET1KA\Desktop\projects\final_project

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Проверка окружения

```powershell
ffmpeg -version
python -m final_project.main --help
pytest -q
```

Проверка NVENC:

```powershell
ffmpeg -encoders | findstr nvenc
```

## Быстрый старт

### GUI

```powershell
python -m final_project.gui
```

### CLI (минимальный пример)

```powershell
python -m final_project.main --input .\video.mp4 --output .\clips --count 5
```

### CLI (несколько входных видео)

```powershell
python -m final_project.main `
  --input .\a.mp4 .\b.mp4 `
  --output .\clips `
  --clip-counts 3 7 `
  --mode speech `
  --min-duration 15 `
  --max-duration 20
```

При пакетной обработке для каждого видео создаются отдельные подпапки в `--output`.

## Параметры CLI

- `--input, -i` - один или несколько входных файлов (обязательный)
- `--output, -o` - папка для результата (обязательный)
- `--mode, -m {speech,beat}` - режим сегментации
- `--count N` - алиас для `--max-clips`
- `--max-clips N` - максимум клипов на входное видео
- `--clip-counts ...` - индивидуальные лимиты клипов (1 значение на все видео или по значению на каждое)
- `--min-duration FLOAT` - минимальная длительность клипа (сек)
- `--max-duration FLOAT` - максимальная длительность клипа (сек)
- `--coords x1 y1 x2 y2` - ручной crop (если не задано, используется стандартный редактор)
- `--blur-radius FLOAT` - радиус размытия
- `--sticker PATH` - путь к стикеру (изображение/GIF/видео)
- `--sticker-size W H` - размер стикера
- `--sticker-position STR` - позиция стикера
- `--sound PATH` - дополнительная аудиодорожка
- `--subtitles / --no-subtitles` - включить/выключить субтитры
- `--subtitle-background / --no-subtitle-background` - фон под субтитрами
- `--subtitle-fontsize N` - размер шрифта субтитров

## Переменные окружения (производительность)

- `FINAL_PROJECT_PARALLEL_EXPORTS` - число параллельных экспортов
- `FINAL_PROJECT_EXPORT_THREADS` - `ffmpeg` threads на один экспорт
- `FINAL_PROJECT_NVENC_PRESET` - preset для `h264_nvenc` (по умолчанию `p6`)
- `FINAL_PROJECT_NVENC_CQ` - качество NVENC (по умолчанию `19`)
- `FINAL_PROJECT_NVENC_RC` - режим bitrate control (по умолчанию `vbr`)
- `FINAL_PROJECT_X264_CRF` - CRF для fallback `libx264` (по умолчанию `18`)

## Сборка Windows-приложения

```powershell
pip install pyinstaller
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

Результат сборки: `dist\FinalProjectVideoCutter\`.

## Тесты

```powershell
pytest -q
```

Покрываются:

- разбор CLI-аргументов
- формирование `ProcessingRequest`
- часть логики сегментации и субтитров
- части генератора и стратегий экспорта

Важно: это не заменяет полноценную end-to-end проверку на реальных медиафайлах.

## Ограничения и замечания

- На целевой машине должен быть доступен `ffmpeg` (если не упакован отдельно).
- Модели распознавания речи (Whisper/faster-whisper) загружаются по мере использования.
- Скорость и качество сегментации зависят от языка, качества звука и фонового шума.
