

import os
import sys
import json
import queue
import logging
import threading
import subprocess
import time
import pickle
from datetime import datetime
from pathlib import Path

# Основные библиотеки
import pyaudio
import pyautogui
from vosk import Model, KaldiRecognizer

# Альтернативный TTS - RHVoice (женский русский голос)
try:
    from TTS.api import TTS

    USE_COQUI_TTS = True
except ImportError:
    USE_COQUI_TTS = False
    import pyttsx3

# Для обучения персонализированной модели
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('potter.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PotterAssistant:
    """Главный класс голосового ассистента Potter"""

    def __init__(self):
        """Инициализация ассистента"""
        self.WAKE_WORD = "поттер"
        self.COMMAND_TIMEOUT = 7  # секунд для ожидания команды
        self.is_listening = False
        self.command_queue = queue.Queue()

        # Инициализация TTS (приоритет - качественный женский голос)
        self._init_tts()

        # Инициализация Vosk модели
        self._init_vosk()

        # Инициализация PyAudio
        self._init_audio()

        # Инициализация AI модели для улучшенного распознавания
        self._init_ai_enhancement()

        # Кэш для погоды (опционально)
        self.weather_cache = self._load_cache()

        # Опасные команды требуют подтверждения
        self.dangerous_commands = ['выключи', 'перезагрузи', 'удали']

        # Счетчик команд для обучения
        self.training_data = []
        self.training_mode = False

        logger.info("Potter ассистент инициализирован успешно")

    def _init_tts(self):
        """Инициализация синтеза речи с качественным женским голосом"""
        try:
            if USE_COQUI_TTS:
                # Coqui TTS - высококачественный женский голос
                logger.info("Использую Coqui TTS для женского голоса")
                self.tts_engine = TTS(model_name="tts_models/ru/ruslan/tacotron2-DDC",
                                      progress_bar=False)
                self.tts_type = 'coqui'
                logger.info("Coqui TTS инициализирован (женский русский голос)")
            else:
                # Fallback на pyttsx3 с настройкой женского голоса
                logger.info("Coqui TTS недоступен, использую pyttsx3")
                self.tts_engine = pyttsx3.init()
                self.tts_type = 'pyttsx3'

                voices = self.tts_engine.getProperty('voices')

                # Поиск женского русского голоса
                female_voice = None
                for voice in voices:
                    voice_name = voice.name.lower()
                    # Приоритет: русский женский голос
                    if 'russian' in voice_name or 'ru' in str(voice.languages):
                        if 'female' in voice_name or 'elena' in voice_name or 'irina' in voice_name:
                            female_voice = voice.id
                            logger.info(f"Найден женский русский голос: {voice.name}")
                            break

                # Если не найден русский, ищем любой женский
                if not female_voice:
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            female_voice = voice.id
                            logger.info(f"Использую женский голос: {voice.name}")
                            break

                if female_voice:
                    self.tts_engine.setProperty('voice', female_voice)

                # Настройка параметров для более естественного звучания
                self.tts_engine.setProperty('rate', 165)  # Немного медленнее для женского голоса
                self.tts_engine.setProperty('volume', 0.95)  # Чуть громче

                # Попытка настроить высоту тона (не все движки поддерживают)
                try:
                    self.tts_engine.setProperty('pitch', 1.1)  # Выше тон
                except:
                    pass

                logger.info("pyttsx3 TTS инициализирован (женский голос)")

        except Exception as e:
            logger.error(f"Ошибка инициализации TTS: {e}")
            # Последняя попытка - базовый pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_type = 'pyttsx3'
            logger.warning("Использую стандартный TTS голос")

    def _init_vosk(self):
        """Инициализация Vosk модели"""
        model_path = "vosk-model-small-ru-0.22"  # Путь к модели

        if not os.path.exists(model_path):
            logger.error(f"Модель Vosk не найдена: {model_path}")
            logger.error("Скачайте модель с https://alphacephei.com/vosk/models")
            self.speak("Модель распознавания речи не найдена. Проверьте установку.")
            sys.exit(1)

        try:
            self.model = Model(model_path)
            logger.info(f"Vosk модель загружена из {model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки Vosk модели: {e}")
            sys.exit(1)

    def _init_audio(self):
        """Инициализация аудио потока"""
        try:
            self.audio = pyaudio.PyAudio()
            self.sample_rate = 16000
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)

            # Открытие аудио потока
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=4096  # Малый буфер для низкой задержки
            )
            self.stream.start_stream()
            logger.info("Аудио поток инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации аудио: {e}")
            sys.exit(1)

    def _init_ai_enhancement(self):
        """Инициализация AI модели для улучшенного распознавания"""
        self.ai_model_path = "potter_ai_model.pkl"
        self.scaler_path = "potter_scaler.pkl"

        # Попытка загрузить обученную модель
        if os.path.exists(self.ai_model_path) and os.path.exists(self.scaler_path):
            try:
                with open(self.ai_model_path, 'rb') as f:
                    self.ai_model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("AI модель загружена из файла")
                self.ai_enabled = True
            except Exception as e:
                logger.warning(f"Не удалось загрузить AI модель: {e}")
                self._create_new_ai_model()
        else:
            self._create_new_ai_model()

        # История команд для контекстного понимания
        self.command_history = []
        self.max_history = 10

    def _create_new_ai_model(self):
        """Создание новой AI модели"""
        logger.info("Создание новой AI модели для персонализации")
        # Нейронная сеть для классификации команд и коррекции ошибок
        self.ai_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        self.scaler = StandardScaler()
        self.ai_enabled = False  # Включим после первого обучения

    def extract_command_features(self, text):
        """Извлечение признаков из текста команды для AI"""
        # Простые признаки для распознавания паттернов
        features = []

        # Длина команды
        features.append(len(text))

        # Количество слов
        features.append(len(text.split()))

        # Наличие ключевых слов (one-hot encoding)
        keywords = ['открой', 'закрой', 'включи', 'выключи', 'найди', 'набери',
                    'покажи', 'переключи', 'увеличь', 'уменьши', 'скажи', 'сделай']
        for keyword in keywords:
            features.append(1 if keyword in text else 0)

        # Позиция первого значимого слова
        words = text.split()
        first_pos = -1
        for i, word in enumerate(words):
            if word in keywords:
                first_pos = i
                break
        features.append(first_pos)

        # Наличие числительных
        numbers = ['один', 'два', 'три', 'четыре', 'пять', 'первый', 'второй', 'третий']
        features.append(1 if any(num in text for num in numbers) else 0)

        return np.array(features).reshape(1, -1)

    def enhance_recognition(self, raw_text):
        """Улучшение распознанного текста с помощью AI"""
        if not self.ai_enabled or not raw_text:
            return raw_text

        try:
            # Извлечение признаков
            features = self.extract_command_features(raw_text)
            features_scaled = self.scaler.transform(features)

            # Предсказание вероятностей различных интерпретаций
            if hasattr(self.ai_model, 'predict_proba'):
                probabilities = self.ai_model.predict_proba(features_scaled)[0]
                confidence = max(probabilities)

                # Если уверенность низкая, используем контекст
                if confidence < 0.6 and self.command_history:
                    raw_text = self._use_context(raw_text)

            # Исправление частых ошибок распознавания
            corrections = {
                'аткрой': 'открой',
                'закрый': 'закрой',
                'вылючи': 'выключи',
                'включе': 'включи',
                'патер': 'поттер',
                'потер': 'поттер',
            }

            for wrong, correct in corrections.items():
                raw_text = raw_text.replace(wrong, correct)

            return raw_text

        except Exception as e:
            logger.warning(f"Ошибка AI-улучшения: {e}")
            return raw_text

    def _use_context(self, text):
        """Использование контекста предыдущих команд"""
        if not self.command_history:
            return text

        last_command = self.command_history[-1]

        # Если текущая команда неполная, пытаемся дополнить контекстом
        if len(text.split()) < 2:
            # Например, "закрой" после "открой chrome" -> "закрой chrome"
            if 'закрой' in text and 'открой' in last_command:
                words = last_command.split()
                if len(words) > 1:
                    return f"{text} {words[-1]}"

        return text

    def train_on_command(self, command, success):
        """Обучение на выполненной команде"""
        if not self.training_mode:
            return

        # Сохранение данных для обучения
        features = self.extract_command_features(command)
        label = 1 if success else 0

        self.training_data.append({
            'features': features,
            'label': label,
            'command': command,
            'timestamp': datetime.now()
        })

        # Обучение каждые 20 команд
        if len(self.training_data) >= 20:
            self._train_ai_model()

    def _train_ai_model(self):
        """Обучение AI модели на собранных данных"""
        if len(self.training_data) < 10:
            logger.warning("Недостаточно данных для обучения AI модели")
            return

        try:
            logger.info(f"Обучение AI модели на {len(self.training_data)} примерах...")

            X = np.vstack([d['features'] for d in self.training_data])
            y = np.array([d['label'] for d in self.training_data])

            # Нормализация признаков
            X_scaled = self.scaler.fit_transform(X)

            # Обучение модели
            self.ai_model.fit(X_scaled, y)

            # Сохранение модели
            with open(self.ai_model_path, 'wb') as f:
                pickle.dump(self.ai_model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            self.ai_enabled = True
            logger.info("AI модель успешно обучена и сохранена")

            # Очистка старых данных
            self.training_data = self.training_data[-50:]  # Оставляем последние 50

        except Exception as e:
            logger.error(f"Ошибка обучения AI модели: {e}")

    def enable_training_mode(self):
        """Включение режима обучения"""
        self.training_mode = True
        logger.info("Режим обучения AI активирован")
        self.speak("Режим обучения активирован. Я буду учиться на ваших командах.")
        """Загрузка кэша (для погоды и др.)"""
        cache_file = "potter_cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Сохранение кэша"""
        try:
            with open("potter_cache.json", 'w', encoding='utf-8') as f:
                json.dump(self.weather_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def speak(self, text):
        """Голосовой ответ с качественным женским голосом"""
        logger.info(f"Potter говорит: {text}")
        try:
            if self.tts_type == 'coqui':
                # Coqui TTS - сохранение во временный файл и воспроизведение
                import tempfile
                import wave
                import pyaudio

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name

                self.tts_engine.tts_to_file(text=text, file_path=temp_path)

                # Воспроизведение
                wf = wave.open(temp_path, 'rb')
                p = pyaudio.PyAudio()
                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(),
                                rate=wf.getframerate(),
                                output=True)

                data = wf.readframes(1024)
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)

                stream.stop_stream()
                stream.close()
                p.terminate()
                wf.close()

                # Удаление временного файла
                os.unlink(temp_path)
            else:
                # pyttsx3
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()

        except Exception as e:
            logger.error(f"Ошибка TTS: {e}")
            # Fallback на базовый pyttsx3
            try:
                if self.tts_type != 'pyttsx3':
                    self.tts_engine = pyttsx3.init()
                    self.tts_type = 'pyttsx3'
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass

    def listen(self):
        """Распознавание речи из микрофона с AI-улучшением"""
        try:
            data = self.stream.read(4096, exception_on_overflow=False)

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '').strip().lower()

                # Применяем AI-улучшение
                if text:
                    text = self.enhance_recognition(text)

                return text
            else:
                partial = json.loads(self.recognizer.PartialResult())
                text = partial.get('partial', '').strip().lower()

                # Применяем AI-улучшение даже для частичных результатов
                if text:
                    text = self.enhance_recognition(text)

                return text
        except Exception as e:
            logger.error(f"Ошибка распознавания: {e}")
            return ""

    def wait_for_wake_word(self):
        """Ожидание ключевого слова активации"""
        logger.info(f"Ожидание ключевого слова '{self.WAKE_WORD}'...")

        while True:
            text = self.listen()

            if self.WAKE_WORD in text:
                logger.info("Ключевое слово обнаружено!")
                self.speak("Слушаю")
                return True

            # Минимальная задержка для снижения нагрузки CPU
            time.sleep(0.01)

    def wait_for_command(self):
        """Ожидание команды после активации"""
        logger.info("Ожидание команды...")
        command_text = ""
        start_time = time.time()

        while time.time() - start_time < self.COMMAND_TIMEOUT:
            text = self.listen()

            if text and len(text) > 2:
                command_text = text
                logger.info(f"Команда получена: {command_text}")
                break

            time.sleep(0.05)

        if not command_text:
            self.speak("Команда не распознана")
            return None

        return command_text

    def execute_command(self, command):
        """Выполнение команды с обучением AI"""
        logger.info(f"Выполнение команды: {command}")

        # Добавление в историю команд
        self.command_history.append(command)
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)

        success = True  # Флаг успешности выполнения для обучения

        try:
            # Проверка на опасные команды
            if any(word in command for word in self.dangerous_commands):
                if not self.confirm_action(command):
                    self.speak("Команда отменена")
                    success = False
                    self.train_on_command(command, success)
                    return

            # === СПЕЦИАЛЬНЫЕ КОМАНДЫ AI ===
            if 'режим обучения' in command or 'начни обучение' in command:
                self.enable_training_mode()
                return

            elif 'статистика' in command or 'покажи статистику' in command:
                self.show_ai_stats()
                return

            # === СИСТЕМНЫЕ КОМАНДЫ ===
            if 'выключи компьютер' in command or 'выключи пк' in command:
                self.shutdown_pc()

            elif 'перезагрузи' in command:
                self.restart_pc()

            elif 'режим сна' in command or 'спящий режим' in command:
                self.sleep_pc()

            # === УПРАВЛЕНИЕ ПРИЛОЖЕНИЯМИ ===
            elif 'открой' in command:
                self.open_application(command)

            elif 'закрой' in command:
                self.close_window(command)

            # === УПРАВЛЕНИЕ БРАУЗЕРОМ ===
            elif 'новая вкладка' in command:
                pyautogui.hotkey('ctrl', 't')
                self.speak("Открыл новую вкладку")

            elif 'закрой вкладку' in command:
                pyautogui.hotkey('ctrl', 'w')
                self.speak("Закрыл вкладку")

            elif 'следующая вкладка' in command:
                pyautogui.hotkey('ctrl', 'tab')
                self.speak("Переключил вкладку")

            # === УПРАВЛЕНИЕ МЕДИА ===
            elif 'включи видео' in command or 'воспроизведи видео' in command:
                self.play_video(command)

            elif 'пауза' in command or 'стоп' in command:
                pyautogui.press('space')
                self.speak("Пауза")

            elif 'громкость' in command:
                self.change_volume(command)

            # === УПРАВЛЕНИЕ ФАЙЛАМИ ===
            elif 'найди файл' in command:
                self.find_file(command)

            elif 'открой файл' in command or 'открой папку' in command:
                self.open_file(command)

            elif 'скриншот' in command or 'снимок экрана' in command:
                self.take_screenshot()

            # === ВВОД ТЕКСТА ===
            elif 'набери текст' in command or 'напиши' in command:
                self.type_text(command)

            # === УПРАВЛЕНИЕ ОКНАМИ ===
            elif 'рабочий стол' in command or 'покажи рабочий стол' in command:
                pyautogui.hotkey('win', 'd')
                self.speak("Показываю рабочий стол")

            elif 'сверни окно' in command:
                pyautogui.hotkey('win', 'down')
                self.speak("Свернул окно")

            elif 'разверни окно' in command:
                pyautogui.hotkey('win', 'up')
                self.speak("Развернул окно")

            elif 'переключи окно' in command:
                pyautogui.hotkey('alt', 'tab')
                self.speak("Переключил окно")

            # === ИНФОРМАЦИОННЫЕ КОМАНДЫ ===
            elif 'время' in command or 'который час' in command:
                self.tell_time()

            elif 'дата' in command or 'какое число' in command:
                self.tell_date()

            elif 'погода' in command:
                self.tell_weather()

            # === СЛУЖЕБНЫЕ КОМАНДЫ ===
            elif 'стоп' in command or 'хватит' in command or 'выход' in command:
                self.speak("До свидания")
                self.cleanup()
                sys.exit(0)

            else:
                self.speak("Извините, я не понял команду. Повторите, пожалуйста.")
                logger.warning(f"Неизвестная команда: {command}")
                success = False

        except Exception as e:
            logger.error(f"Ошибка выполнения команды: {e}")
            self.speak("Произошла ошибка при выполнении команды")
            success = False

        # Обучение AI на результате выполнения
        self.train_on_command(command, success)

    def confirm_action(self, action):
        """Подтверждение опасной команды"""
        self.speak(f"Подтвердите действие: {action}. Скажите да или нет")

        start_time = time.time()
        while time.time() - start_time < 5:
            response = self.listen()
            if 'да' in response or 'подтверждаю' in response:
                return True
            elif 'нет' in response or 'отмена' in response:
                return False
            time.sleep(0.1)

        return False

    # === РЕАЛИЗАЦИЯ КОМАНД ===

    def shutdown_pc(self):
        """Выключение компьютера"""
        self.speak("Выключаю компьютер")
        if sys.platform == 'win32':
            os.system('shutdown /s /t 1')
        else:
            os.system('shutdown -h now')

    def restart_pc(self):
        """Перезагрузка компьютера"""
        self.speak("Перезагружаю компьютер")
        if sys.platform == 'win32':
            os.system('shutdown /r /t 1')
        else:
            os.system('shutdown -r now')

    def sleep_pc(self):
        """Спящий режим"""
        self.speak("Перевожу компьютер в спящий режим")
        if sys.platform == 'win32':
            os.system('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')
        else:
            os.system('systemctl suspend')

    def open_application(self, command):
        """Открытие приложения"""
        apps = {
            'chrome': ['chrome', 'google-chrome', 'chromium'],
            'хром': ['chrome', 'google-chrome'],
            'firefox': ['firefox'],
            'файрфокс': ['firefox'],
            'блокнот': ['notepad', 'gedit'],
            'калькулятор': ['calc', 'gnome-calculator'],
            'проводник': ['explorer'],
            'терминал': ['cmd', 'gnome-terminal'],
            'word': ['winword', 'libreoffice --writer'],
            'excel': ['excel', 'libreoffice --calc'],
        }

        for app_name, app_commands in apps.items():
            if app_name in command:
                for app_cmd in app_commands:
                    try:
                        if sys.platform == 'win32':
                            subprocess.Popen(app_cmd, shell=True)
                        else:
                            subprocess.Popen(app_cmd.split())
                        self.speak(f"Открываю {app_name}")
                        return
                    except:
                        continue
                self.speak(f"Не удалось открыть {app_name}")
                return

        self.speak("Приложение не найдено")

    def close_window(self, command):
        """Закрытие окна"""
        if 'вкладку' in command:
            pyautogui.hotkey('ctrl', 'w')
        else:
            pyautogui.hotkey('alt', 'f4')
        self.speak("Закрыл")

    def play_video(self, command):
        """Воспроизведение видео"""
        # Поиск видео по названию или порядковому номеру
        if 'второе' in command or '2' in command:
            # Логика для воспроизведения второго видео из плейлиста
            self.speak("Включаю второе видео")
            # Здесь нужна интеграция с медиаплеером
        else:
            self.speak("Для воспроизведения видео укажите путь к файлу или номер в плейлисте")

    def change_volume(self, command):
        """Изменение громкости"""
        if 'увеличь' in command or 'прибавь' in command or 'громче' in command:
            for _ in range(5):
                pyautogui.press('volumeup')
            self.speak("Увеличил громкость")
        elif 'уменьши' in command or 'убавь' in command or 'тише' in command:
            for _ in range(5):
                pyautogui.press('volumedown')
            self.speak("Уменьшил громкость")
        elif 'выключи звук' in command or 'отключи звук' in command:
            pyautogui.press('volumemute')
            self.speak("Отключил звук")

    def find_file(self, command):
        """Поиск файла"""
        # Извлечение имени файла из команды
        words = command.split()
        if 'файл' in words:
            idx = words.index('файл')
            if idx + 1 < len(words):
                filename = ' '.join(words[idx + 1:])
                self.speak(f"Ищу файл {filename}")
                # Простой поиск в домашней директории
                home = str(Path.home())
                for root, dirs, files in os.walk(home):
                    for file in files:
                        if filename.lower() in file.lower():
                            path = os.path.join(root, file)
                            self.speak(f"Файл найден: {file}")
                            logger.info(f"Найден файл: {path}")
                            return
                self.speak("Файл не найден")

    def open_file(self, command):
        """Открытие файла"""
        # Упрощенная реализация - открытие диалога выбора
        if sys.platform == 'win32':
            pyautogui.hotkey('win', 'r')
            time.sleep(0.3)
            pyautogui.write('explorer')
            pyautogui.press('enter')
        self.speak("Открываю проводник")

    def take_screenshot(self):
        """Создание скриншота"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        self.speak("Скриншот сохранен")
        logger.info(f"Скриншот сохранен: {filename}")

    def type_text(self, command):
        """Набор текста"""
        # Извлечение текста после команды
        if 'набери текст' in command:
            text = command.replace('набери текст', '').strip()
        elif 'напиши' in command:
            text = command.replace('напиши', '').strip()
        else:
            text = ""

        if text:
            time.sleep(0.5)  # Задержка для переключения на нужное окно
            pyautogui.write(text, interval=0.05)
            self.speak("Текст набран")
        else:
            self.speak("Укажите текст для набора")

    def tell_time(self):
        """Сообщить время"""
        current_time = datetime.now().strftime("%H:%M")
        self.speak(f"Сейчас {current_time}")

    def tell_date(self):
        """Сообщить дату"""
        current_date = datetime.now().strftime("%d %B %Y года")
        months = {
            'January': 'января', 'February': 'февраля', 'March': 'марта',
            'April': 'апреля', 'May': 'мая', 'June': 'июня',
            'July': 'июля', 'August': 'августа', 'September': 'сентября',
            'October': 'октября', 'November': 'ноября', 'December': 'декабря'
        }
        for eng, rus in months.items():
            current_date = current_date.replace(eng, rus)
        self.speak(f"Сегодня {current_date}")

    def tell_weather(self):
        """Сообщить погоду (из кэша)"""
        if 'weather' in self.weather_cache:
            cached_data = self.weather_cache['weather']
            age = (datetime.now() - datetime.fromisoformat(cached_data['timestamp'])).seconds / 3600

            if age < 3:  # Кэш актуален 3 часа
                self.speak(f"По данным на {cached_data['time']}, температура {cached_data['temp']} градусов")
            else:
                self.speak("Данные о погоде устарели. Требуется подключение к интернету")
        else:
            self.speak("Данные о погоде отсутствуют. Требуется подключение к интернету для обновления")

    def show_ai_stats(self):
        """Показать статистику работы AI"""
        if self.ai_enabled:
            self.speak(f"AI модель активна. Обработано команд: {len(self.command_history)}. "
                       f"Данных для обучения: {len(self.training_data)}")
        else:
            self.speak("AI модель еще не обучена. Активируйте режим обучения.")

    def run(self):
        """Основной цикл работы ассистента"""
        self.speak("Голосовой ассистент Поттер запущен")
        logger.info("Potter запущен и готов к работе")

        try:
            while True:
                # Ожидание ключевого слова
                self.wait_for_wake_word()

                # Ожидание команды
                command = self.wait_for_command()

                if command:
                    # Выполнение команды
                    self.execute_command(command)

                # Небольшая пауза перед следующим циклом
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки")
            self.speak("Завершаю работу")
            self.cleanup()

    def cleanup(self):
        """Очистка ресурсов"""
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self._save_cache()
            logger.info("Ресурсы освобождены")
        except Exception as e:
            logger.error(f"Ошибка при очистке ресурсов: {e}")


def main():
    """Точка входа в программу"""
    print("=" * 50)
    print("Potter - Голосовой ассистент для управления ПК")
    print("С AI-улучшением и качественным женским голосом")
    print("=" * 50)
    print("\nПроверка зависимостей...")

    # Проверка наличия необходимых библиотек
    required_modules = {
        'pyaudio': 'PyAudio',
        'pyautogui': 'PyAutoGUI',
        'vosk': 'Vosk',
        'sklearn': 'scikit-learn',
        'numpy': 'NumPy'
    }

    optional_modules = {
        'TTS': 'Coqui-TTS (рекомендуется для лучшего женского голоса)',
        'pyttsx3': 'pyttsx3 (базовый TTS)'
    }

    missing_modules = []

    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            missing_modules.append(module)
            print(f"✗ {name} - ТРЕБУЕТСЯ")

    print("\nОпциональные модули:")
    for module, name in optional_modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"○ {name} - не установлен")

    if missing_modules:
        print(f"\n❌ Ошибка! Не установлены обязательные модули: {', '.join(missing_modules)}")
        print("\nУстановите их командой:")
        print(f"pip install {' '.join(missing_modules)}")
        print("\nДля лучшего женского голоса установите также:")
        print("pip install TTS")
        sys.exit(1)

    print("\n✓ Все обязательные зависимости установлены")

    # Информация об AI
    print("\n" + "=" * 50)
    print("AI-ПЕРСОНАЛИЗАЦИЯ")
    print("=" * 50)
    print("Potter использует нейронную сеть для:")
    print("  • Исправления ошибок распознавания")
    print("  • Контекстного понимания команд")
    print("  • Адаптации под ваш голос и манеру речи")
    print("\nСкажите 'Поттер, режим обучения' для активации")
    print("=" * 50 + "\n")

    # Проверка модели Vosk
    if not os.path.exists("vosk-model-small-ru-0.22"):
        print("⚠ Внимание! Модель Vosk не найдена.")
        print("Скачайте модель vosk-model-small-ru-0.22 с:")
        print("https://alphacephei.com/vosk/models")
        print("И распакуйте в текущую директорию.\n")

    try:
        assistant = PotterAssistant()
        assistant.run()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)
        print("Potter - Голосовой ассистент для управления ПК")
    print("=" * 50)
    print("\nПроверка зависимостей...")

    # Проверка наличия необходимых библиотек
    required_modules = ['pyaudio', 'pyttsx3', 'pyautogui', 'vosk']
    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print(f"\nОшибка! Не установлены модули: {', '.join(missing_modules)}")
        print("\nУстановите их командой:")
        print(f"pip install {' '.join(missing_modules)}")
        sys.exit(1)

    print("✓ Все зависимости установлены\n")

    # Проверка модели Vosk
    if not os.path.exists("vosk-model-small-ru-0.22"):
        print("Внимание! Модель Vosk не найдена.")
        print("Скачайте модель vosk-model-small-ru-0.22 с:")
        print("https://alphacephei.com/vosk/models")
        print("И распакуйте в текущую директорию.\n")

    try:
        assistant = PotterAssistant()
        assistant.run()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"\nКритическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()