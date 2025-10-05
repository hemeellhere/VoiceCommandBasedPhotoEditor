import sys
import os
import re
import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
from typing import Optional, List
from fuzzywuzzy import fuzz
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QEventLoop
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QKeySequence, QAction, QPalette, QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QStatusBar,
    QPushButton, QFileDialog, QToolBar, QSizePolicy, QDockWidget,
    QMessageBox, QDialog, QScrollArea, QSlider, QProgressBar,
    QListWidget, QGroupBox, QMenu, QTextEdit
)

# ========== Simple Theme Colors ==========

class Theme:
    """Windows 11 dark theme - simple colors"""
    # Main backgrounds
    BG_PRIMARY = "#202020"
    BG_SECONDARY = "#2B2B2B"
    BG_TERTIARY = "#323232"
    
    # Accent colors for different button types
    ACCENT_BLUE = "#0078D4"      # Windows signature blue
    ACCENT_GREEN = "#107C10"     # Success/positive actions
    ACCENT_RED = "#C42B1C"       # Danger/destructive actions
    ACCENT_ORANGE = "#FFA500"    # Warning/attention
    
    # Text colors
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#E0E0E0"
    
    BORDER = "#444444"

# ========== Utilities ==========

def clamp(val: int, lo: int, hi: int) -> int:
    """Clamp value between min and max - prevents slider values from going out of range"""
    return max(lo, min(hi, val))

def numpy_to_qimage(img: np.ndarray) -> QImage:
    """Convert OpenCV's NumPy array to Qt's QImage format
    
    OpenCV uses BGR format and NumPy arrays, while Qt uses RGB and QImage.
    This function handles the conversion and different image types (grayscale, RGB, RGBA).
    """
    if img is None or img.size == 0:
        return QImage()
    
    # Ensure image data is in uint8 format (0-255)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
    
    # Handle grayscale images
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
    
    # Handle color images
    if img.ndim == 3:
        h, w, ch = img.shape
        if ch == 3:
            # Convert BGR (OpenCV) to RGB (Qt)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        if ch == 4:
            # Handle images with alpha channel
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            return QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    
    # Fallback: convert to grayscale if format is unknown
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    return QImage(gray.data, w, h, w, QImage.Format_Grayscale8).copy()

def compress_image(img: np.ndarray) -> bytes:
    """Compress image to JPEG bytes for undo stack
    
    Storing full images in undo stack uses too much memory.
    JPEG compression reduces memory usage while maintaining acceptable quality.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, buffer = cv2.imencode('.jpg', img, encode_param)
    return buffer.tobytes()

def decompress_image(buffer: bytes) -> np.ndarray:
    """Decompress JPEG bytes back to image array"""
    return cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED)

# ========== Speech Threads ==========

class SpeechThread(QThread):
    """Background thread for voice recognition
    
    Runs in separate thread to prevent UI freezing during voice input.
    Uses Google's speech recognition API.
    """
    command_signal = Signal(str)      # Emits recognized text
    error_signal = Signal(str)        # Emits error messages
    status_signal = Signal(str)       # Emits status updates (listening, processing, etc.)

    def __init__(self, timeout: float = 5.0, phrase_time_limit: float = 5.0):
        super().__init__()
        self.recognizer = sr.Recognizer()
        # Adjust these values if microphone is too sensitive/insensitive
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.microphone = None
        self.running = False
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit

    def run(self):
        """Main thread loop - continuously listens for voice commands"""
        # Initialize microphone
        try:
            self.microphone = sr.Microphone()
            self.status_signal.emit("Ready")
        except Exception as e:
            self.error_signal.emit(f"Microphone error: {str(e)[:50]}")
            return

        # Calibrate for ambient noise (important for accuracy)
        try:
            with self.microphone as source:
                self.status_signal.emit("Calibrating...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.status_signal.emit("Ready")
        except Exception as e:
            self.error_signal.emit(f"Calibration failed: {str(e)[:50]}")
            return

        # Main listening loop
        while self.running:
            try:
                with self.microphone as source:
                    self.status_signal.emit("Listening...")
                    # Listen for audio input with timeout
                    audio = self.recognizer.listen(source, timeout=self.timeout, phrase_time_limit=self.phrase_time_limit)
                
                self.status_signal.emit("Processing...")
                # Send audio to Google for recognition
                text = self.recognizer.recognize_google(audio, show_all=False).lower()
                self.command_signal.emit(text)
                
            except sr.WaitTimeoutError:
                # No speech detected within timeout - just continue listening
                continue
            except sr.UnknownValueError:
                # Speech detected but couldn't understand it
                self.error_signal.emit("Could not understand")
            except sr.RequestError as e:
                # Problem with Google API (network issue, etc.)
                self.error_signal.emit(f"API error: {str(e)[:50]}")
                self.msleep(2000)  # Wait before retrying
            except Exception as e:
                # Catch-all for unexpected errors
                self.error_signal.emit(f"Error: {str(e)[:50]}")
                self.msleep(1000)

    def start_listening(self):
        """Start the voice recognition thread"""
        self.running = True
        if not self.isRunning():
            self.start()

    def stop_listening(self):
        """Stop the voice recognition thread"""
        self.running = False
        self.wait(1000)


class TTSWorker(QThread):
    """Text-to-speech worker thread
    
    Runs TTS in background to avoid blocking UI.
    Uses pyttsx3 which works offline.
    """
    say_signal = Signal(str)  # Signal to speak text

    def __init__(self, rate: int = 160):
        super().__init__()
        self._engine = None
        self._rate = rate  # Speaking speed (words per minute)
        self._running = True

    def run(self):
        """Initialize TTS engine and process speak requests"""
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self._rate)
            self._engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"TTS init error: {e}")
            return

        # Connect signal to speak method
        self.say_signal.connect(self._on_say, Qt.QueuedConnection)
        
        # Keep thread alive to process speak requests
        while self._running:
            QApplication.processEvents(QEventLoop.ProcessEventsFlags.AllEvents, 10)
            self.msleep(50)
        
        self._cleanup()

    @Slot(str)
    def _on_say(self, text: str):
        """Actually speak the text"""
        if not self._engine or not self._running:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

    def _cleanup(self):
        """Clean up TTS engine on exit"""
        try:
            if self._engine:
                self._engine.stop()
        except Exception:
            pass

    def stop(self):
        """Stop the TTS thread"""
        self._running = False
        try:
            if self._engine:
                self._engine.stop()
        except Exception:
            pass
        self.wait(2000)


class ImageProcessingThread(QThread):
    """Thread for heavy image processing operations
    
    Some operations like super resolution can take several seconds.
    Running them in a separate thread keeps the UI responsive.
    """
    result_signal = Signal(np.ndarray)   # Emits processed image
    progress_signal = Signal(int)        # Emits progress percentage
    error_signal = Signal(str)           # Emits error message

    def __init__(self, operation, *args):
        super().__init__()
        self.operation = operation  # The image processing function
        self.args = args           # Arguments for the function

    def run(self):
        """Execute the image processing operation"""
        try:
            self.progress_signal.emit(30)
            result = self.operation(*self.args)
            self.progress_signal.emit(90)
            self.result_signal.emit(result)
            self.progress_signal.emit(100)
        except Exception as e:
            self.error_signal.emit(str(e))


# ========== Help Dialog ==========

class HelpDialog(QDialog):
    """Simple help dialog showing keyboard shortcuts and voice commands"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help & Documentation")
        self.setMinimumSize(650, 700)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("📚 Help & Documentation")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Content
        content = QTextEdit()
        content.setReadOnly(True)
        content.setHtml(self._get_help_content())
        layout.addWidget(content)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setFixedHeight(35)
        layout.addWidget(close_btn)
        
    def _get_help_content(self):
        """Generate HTML content for help dialog"""
        return """
        <h3>⌨️ Keyboard Shortcuts</h3>
        <table width='100%'>
            <tr><td width='30%'><b>Ctrl+O</b></td><td>Open Image</td></tr>
            <tr><td><b>Ctrl+S</b></td><td>Save Image</td></tr>
            <tr><td><b>Ctrl+Z</b></td><td>Undo</td></tr>
            <tr><td><b>Ctrl+Y</b></td><td>Redo</td></tr>
            <tr><td><b>Ctrl++</b></td><td>Zoom In</td></tr>
            <tr><td><b>Ctrl+-</b></td><td>Zoom Out</td></tr>
            <tr><td><b>Ctrl+0</b></td><td>Reset Zoom</td></tr>
            <tr><td><b>F</b></td><td>Fit to Window</td></tr>
            <tr><td><b>F1</b></td><td>Show This Help</td></tr>
        </table>

        <h3>🎤 Voice Commands</h3>
        <p><b>Filters:</b> grayscale, blur, sharpen, edge detection, sepia, invert, histogram equalization, adaptive threshold</p>
        <p><b>Adjustments:</b> brightness by [X], contrast by [X], saturation by [X], hue by [X]</p>
        <p><b>Transforms:</b> rotate left, rotate right, flip horizontal, flip vertical</p>
        <p><b>Navigation:</b> zoom in, zoom out, reset zoom, fit</p>
        <p><b>History:</b> undo, redo, reset image</p>
        <p><b>App:</b> help, exit</p>
        """


# ========== Main Window ==========

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Photo Editor Pro - Voice Control")
        self.setGeometry(100, 100, 1300, 800)
        self.setMinimumSize(1000, 600)
        
        # Image state variables
        self.current_image: Optional[np.ndarray] = None  # Currently displayed image
        self.original_image: Optional[np.ndarray] = None  # Original loaded image (for reset)
        self.zoom_level: float = 1.0
        
        # Undo/Redo stacks (stores compressed images)
        self.undo_stack: List[bytes] = []
        self.redo_stack: List[bytes] = []
        self.max_stack_size: int = 20  # Limit to prevent excessive memory usage
        
        # File management
        self.has_unsaved_changes = False
        self.recent_files: List[str] = []
        self.max_recent_files: int = 10
        
        # Voice command history
        self.command_history: List[str] = []
        self.max_history_size: int = 15
        
        # Setup everything
        self._apply_theme()
        self._setup_ui()
        self._setup_threads()
        self._compile_patterns()
        
        # Enable Ctrl+Wheel zoom on image
        self.image_label.installEventFilter(self)
        
        # Cleanup threads on exit
        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(self._cleanup)

    def _apply_theme(self):
        """Apply Windows 11 dark theme colors and styles"""
        QApplication.setStyle("Fusion")
        
        # Set color palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(Theme.BG_PRIMARY))
        palette.setColor(QPalette.WindowText, QColor(Theme.TEXT_PRIMARY))
        palette.setColor(QPalette.Base, QColor(Theme.BG_SECONDARY))
        palette.setColor(QPalette.AlternateBase, QColor(Theme.BG_TERTIARY))
        palette.setColor(QPalette.Text, QColor(Theme.TEXT_PRIMARY))
        palette.setColor(QPalette.Button, QColor(Theme.BG_TERTIARY))
        palette.setColor(QPalette.ButtonText, QColor(Theme.TEXT_PRIMARY))
        palette.setColor(QPalette.Highlight, QColor(Theme.ACCENT_BLUE))
        palette.setColor(QPalette.HighlightedText, QColor(Theme.TEXT_PRIMARY))
        QApplication.setPalette(palette)
        
        # Apply global stylesheet for consistent look
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {Theme.BG_PRIMARY};
            }}
            QPushButton {{
                background-color: {Theme.ACCENT_BLUE};
                color: {Theme.TEXT_PRIMARY};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 11px;
                min-height: 28px;
            }}
            QPushButton:hover {{
                background-color: #1984D8;
            }}
            QPushButton:pressed {{
                background-color: #005A9E;
            }}
            QPushButton:checked {{
                background-color: {Theme.ACCENT_GREEN};
            }}
            QGroupBox {{
                background: {Theme.BG_SECONDARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {Theme.BG_TERTIARY};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {Theme.ACCENT_BLUE};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {Theme.ACCENT_BLUE};
                border-radius: 2px;
            }}
            QProgressBar {{
                background: {Theme.BG_TERTIARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 4px;
                text-align: center;
                color: {Theme.TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background: {Theme.ACCENT_BLUE};
                border-radius: 4px;
            }}
            QListWidget {{
                background: {Theme.BG_SECONDARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 4px;
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 6px;
                border-radius: 3px;
            }}
            QListWidget::item:hover {{
                background: {Theme.BG_TERTIARY};
            }}
            QListWidget::item:selected {{
                background: {Theme.ACCENT_BLUE};
            }}
            QDockWidget::title {{
                background: {Theme.BG_TERTIARY};
                padding: 6px;
                font-weight: bold;
            }}
            QToolBar {{
                background: {Theme.BG_SECONDARY};
                border-bottom: 1px solid {Theme.BORDER};
                spacing: 6px;
                padding: 6px;
            }}
            QStatusBar {{
                background: {Theme.BG_SECONDARY};
                border-top: 1px solid {Theme.BORDER};
            }}
        """)

    def _setup_ui(self):
        """Setup all UI components in order"""
        self._setup_status_bar()
        self._setup_central_widget()
        self._setup_toolbar()
        self._setup_sidebar()
        self._setup_adjustments_dock()
        self._setup_history_dock()
        self._setup_actions()

    def _setup_status_bar(self):
        """Setup bottom status bar with progress indicator"""
        self.status_bar = QStatusBar()
        
        # Status message label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 0 10px;")
        self.status_bar.addWidget(self.status_label)
        
        # Spacer to push progress bar to the right
        self.status_bar.addPermanentWidget(QLabel(""))
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.setStatusBar(self.status_bar)

    def _setup_central_widget(self):
        """Setup main image display area"""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Scrollable area for zoomed images
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setStyleSheet(f"border: none; background: {Theme.BG_PRIMARY};")
        
        # Label to display image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setAcceptDrops(True)  # Enable drag & drop
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._reset_image_label()
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

    def _reset_image_label(self):
        """Reset image label to show placeholder text"""
        self.image_label.setText(
            "📸 Drop Image Here\n\n"
            "or use Import button / Voice commands\n\n"
            "Press F1 for Help"
        )
        self.image_label.setStyleSheet(f"""
            background: {Theme.BG_SECONDARY};
            border: 2px dashed {Theme.BORDER};
            border-radius: 8px;
            color: #888;
            font-size: 14px;
            padding: 40px;
        """)

    def _setup_toolbar(self):
        """Setup top toolbar with main action buttons"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        
        # Voice control button (checkable - stays pressed when active)
        self.voice_button = QPushButton("🎤 Voice Control")
        self.voice_button.setCheckable(True)
        self.voice_button.clicked.connect(self.toggle_voice)
        toolbar.addWidget(self.voice_button)
        
        toolbar.addSeparator()
        
        # File operations
        import_btn = QPushButton("📁 Import")
        import_btn.clicked.connect(self.load_image_dialog)
        toolbar.addWidget(import_btn)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_image_dialog)
        toolbar.addWidget(save_btn)
        
        # Recent files dropdown menu
        self.recent_menu = QMenu(self)
        recent_btn = QPushButton("📋 Recent")
        recent_btn.clicked.connect(lambda: self.recent_menu.exec(self.cursor().pos()))
        toolbar.addWidget(recent_btn)
        
        toolbar.addSeparator()
        
        # Undo/Redo
        undo_btn = QPushButton("↶ Undo")
        undo_btn.clicked.connect(self.undo)
        toolbar.addWidget(undo_btn)
        
        redo_btn = QPushButton("↷ Redo")
        redo_btn.clicked.connect(self.redo)
        toolbar.addWidget(redo_btn)
        
        toolbar.addSeparator()
        
        # Help
        help_btn = QPushButton("❓ Help")
        help_btn.clicked.connect(self.show_help)
        toolbar.addWidget(help_btn)

    def _setup_sidebar(self):
        """Setup left sidebar with filter categories"""
        sidebar = QDockWidget("🎨 Filters & Effects")
        sidebar.setFixedWidth(170)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Scrollable area for all filter buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)
        
        # Organize filters into categories
        categories = {
            "🎭 Basic": [
                ("Grayscale", self.apply_grayscale),
                ("Blur", self.apply_blur),
                ("Sharpen", self.apply_sharpen),
                ("Edge Detect", self.apply_edge_detection),
            ],
            "🌈 Color": [
                ("Sepia", self.apply_sepia),
                ("Invert", self.apply_invert),
                ("Saturation+", self.apply_saturation_default),
            ],
            "✨ Advanced": [
                ("Histogram Eq", self.apply_histogram_equalization),
                ("Adaptive", self.apply_adaptive_thresholding),
            ],
            "🔄 Transform": [
                ("Rotate Left", self.rotate_left),
                ("Rotate Right", self.rotate_right),
                ("Flip H", self.flip_horizontal),
                ("Flip V", self.flip_vertical),
            ],
            "🔍 Zoom": [
                ("Zoom In", self.zoom_in),
                ("Zoom Out", self.zoom_out),
                ("Fit Window", self.fit_to_window),
                ("Reset Zoom", self.reset_zoom),
            ],
        }
        
        # Create group boxes for each category
        for category, filters in categories.items():
            group = QGroupBox(category)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(4)
            
            # Add buttons for each filter in the category
            for name, slot in filters:
                btn = QPushButton(name)
                btn.clicked.connect(slot)
                group_layout.addWidget(btn)
            
            scroll_layout.addWidget(group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        sidebar.setWidget(widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, sidebar)

    def _setup_adjustments_dock(self):
        """Setup right panel with adjustment sliders"""
        dock = QDockWidget("⚙️ Adjustments")
        dock.setFixedWidth(220)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        
        # Brightness slider (-100 to +100)
        bright_group = QGroupBox("💡 Brightness")
        bright_layout = QVBoxLayout(bright_group)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.apply_brightness)
        self.brightness_label = QLabel("0")
        self.brightness_label.setAlignment(Qt.AlignCenter)
        # Update label when slider moves
        self.brightness_slider.valueChanged.connect(lambda v: self.brightness_label.setText(str(v)))
        bright_layout.addWidget(self.brightness_slider)
        bright_layout.addWidget(self.brightness_label)
        layout.addWidget(bright_group)
        
        # Contrast slider (0 to 200, default 100 = normal)
        contrast_group = QGroupBox("🔆 Contrast")
        contrast_layout = QVBoxLayout(contrast_group)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.apply_contrast)
        self.contrast_label = QLabel("100")
        self.contrast_label.setAlignment(Qt.AlignCenter)
        self.contrast_slider.valueChanged.connect(lambda v: self.contrast_label.setText(str(v)))
        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_label)
        layout.addWidget(contrast_group)
        
        # Saturation slider (0 to 200, default 100 = normal)
        saturation_group = QGroupBox("🎨 Saturation")
        saturation_layout = QVBoxLayout(saturation_group)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 200)
        self.saturation_slider.setValue(100)
        self.saturation_slider.valueChanged.connect(self.apply_saturation)
        self.saturation_label = QLabel("100")
        self.saturation_label.setAlignment(Qt.AlignCenter)
        self.saturation_slider.valueChanged.connect(lambda v: self.saturation_label.setText(str(v)))
        saturation_layout.addWidget(self.saturation_slider)
        saturation_layout.addWidget(self.saturation_label)
        layout.addWidget(saturation_group)
        
        # Hue slider (-180 to +180 degrees)
        hue_group = QGroupBox("🌈 Hue")
        hue_layout = QVBoxLayout(hue_group)
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setRange(-180, 180)
        self.hue_slider.setValue(0)
        self.hue_slider.valueChanged.connect(self.apply_hue)
        self.hue_label = QLabel("0")
        self.hue_label.setAlignment(Qt.AlignCenter)
        self.hue_slider.valueChanged.connect(lambda v: self.hue_label.setText(str(v)))
        hue_layout.addWidget(self.hue_slider)
        hue_layout.addWidget(self.hue_label)
        layout.addWidget(hue_group)
        
        # Reset all adjustments button
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.reset_adjustments)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _setup_history_dock(self):
        """Setup command history panel"""
        dock = QDockWidget("📜 Command History")
        dock.setFixedWidth(220)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # List widget to show command history
        self.history_list = QListWidget()
        layout.addWidget(self.history_list)
        
        # Clear history button
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(lambda: self.history_list.clear())
        layout.addWidget(clear_btn)
        
        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _setup_actions(self):
        """Setup keyboard shortcuts"""
        actions = [
            ("Open", QKeySequence.Open, self.load_image_dialog),
            ("Save", QKeySequence.Save, self.save_image_dialog),
            ("Undo", QKeySequence.Undo, self.undo),
            ("Redo", QKeySequence.Redo, self.redo),
            ("Zoom In", QKeySequence.ZoomIn, self.zoom_in),
            ("Zoom Out", QKeySequence.ZoomOut, self.zoom_out),
            ("Reset Zoom", QKeySequence("Ctrl+0"), self.reset_zoom),
            ("Fit", QKeySequence("F"), self.fit_to_window),
            ("Help", QKeySequence.HelpContents, self.show_help),
        ]
        
        # Create and register each action
        for name, shortcut, slot in actions:
            action = QAction(name, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            self.addAction(action)

    def _setup_threads(self):
        """Initialize voice recognition and TTS threads"""
        # Speech recognition thread
        self.speech_thread = SpeechThread(timeout=5.0, phrase_time_limit=5.0)
        self.speech_thread.command_signal.connect(self.handle_command)
        self.speech_thread.error_signal.connect(self.on_speech_error)
        self.speech_thread.status_signal.connect(self.on_speech_status)
        
        # Text-to-speech thread
        self.tts_thread = TTSWorker(rate=165)
        self.tts_thread.start()

    def _compile_patterns(self):
        """Compile regex patterns for parsing voice commands with parameters
        
        These patterns extract numbers from commands like:
        "brightness by 50" -> extracts 50
        "contrast 120" -> extracts 120
        """
        self.re_brightness = re.compile(r"(?:brightness|brighten)\s*(?:by)?\s*(-?\d+)")
        self.re_contrast = re.compile(r"(?:contrast)\s*(?:by)?\s*(-?\d+)")
        self.re_saturation = re.compile(r"(?:saturation|saturate)\s*(?:by)?\s*(-?\d+)")
        self.re_hue = re.compile(r"(?:hue)\s*(?:by)?\s*(-?\d+)")

    # ========== Speech Methods ==========

    def toggle_voice(self, checked: bool):
        """Toggle voice control on/off"""
        if checked:
            # Start listening
            self.voice_button.setText("🎤 Listening...")
            self.speech_thread.start_listening()
            self.speak("Voice control activated")
        else:
            # Stop listening
            self.voice_button.setText("🎤 Voice Control")
            self.speech_thread.stop_listening()
            self.speak("Voice control deactivated")

    def speak(self, text: str):
        """Speak text and show in status bar"""
        self.status(text)
        self.tts_thread.say_signal.emit(text)

    def on_speech_error(self, error: str):
        """Handle speech recognition errors"""
        self.status(f"⚠️ {error}")

    def on_speech_status(self, status: str):
        """Update voice button text with current status"""
        if self.voice_button.isChecked():
            self.voice_button.setText(f"🎤 {status}")

    # ========== File Operations ==========

    def load_image_dialog(self):
        """Show file dialog to select and load an image"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Image", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif);;All Files (*)"
        )
        if filename:
            self._load_image_from_path(filename)

    def _load_image_from_path(self, path: str):
        """Load image from file path with progress indicator"""
        if not os.path.exists(path):
            self.status("❌ File not found")
            return
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            # Load image using OpenCV
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                self.status("❌ Failed to load image")
                return
            
            self.progress_bar.setValue(50)
            
            # Resize if image is too large (prevents memory issues)
            max_size = 2048
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Store original for reset functionality
            self.current_image = img
            self.original_image = img.copy()
            self.zoom_level = 1.0
            
            # Clear history stacks for new image
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.has_unsaved_changes = False
            
            # Add to recent files list
            if path not in self.recent_files:
                self.recent_files.insert(0, path)
                self.recent_files = self.recent_files[:self.max_recent_files]
                self._update_recent_menu()
            
            self.progress_bar.setValue(100)
            self.display_image()
            
            filename = os.path.basename(path)
            self.speak(f"Image loaded: {filename}")
            self.status(f"✓ Loaded: {filename}")
            
        except Exception as e:
            self.status(f"❌ Error: {str(e)[:50]}")
        finally:
            self.progress_bar.setVisible(False)

    def save_image_dialog(self):
        """Show save dialog and save current image"""
        if self.current_image is None:
            self.status("⚠️ No image to save")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", 
            "PNG (*.png);;JPEG (*.jpg);;WebP (*.webp);;BMP (*.bmp)"
        )
        
        if filename:
            try:
                success = cv2.imwrite(filename, self.current_image)
                if success:
                    self.has_unsaved_changes = False
                    self.speak("Image saved")
                    self.status(f"✓ Saved: {os.path.basename(filename)}")
                else:
                    self.status("❌ Save failed")
            except Exception as e:
                self.status(f"❌ Error: {str(e)[:50]}")

    def _update_recent_menu(self):
        """Update the recent files dropdown menu"""
        self.recent_menu.clear()
        
        if not self.recent_files:
            action = self.recent_menu.addAction("No recent files")
            action.setEnabled(False)
            return
        
        # Add each recent file
        for path in self.recent_files:
            filename = os.path.basename(path)
            action = self.recent_menu.addAction(f"📄 {filename}")
            action.triggered.connect(lambda checked, p=path: self._load_image_from_path(p))
        
        # Add clear option
        self.recent_menu.addSeparator()
        clear_action = self.recent_menu.addAction("Clear Recent")
        clear_action.triggered.connect(lambda: setattr(self, 'recent_files', []) or self._update_recent_menu())

    # ========== Drag & Drop ==========

    def dragEnterEvent(self, event):
        """Accept drag events with image files"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle dropped image files"""
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                # Check if file is an image
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.gif')):
                    self._load_image_from_path(path)
                    return

    # ========== Undo/Redo ==========

    def save_image_state(self):
        """Save current image state to undo stack before making changes"""
        if self.current_image is not None:
            # Compress and store current state
            compressed = compress_image(self.current_image)
            self.undo_stack.append(compressed)
            
            # Limit stack size to prevent memory issues
            if len(self.undo_stack) > self.max_stack_size:
                self.undo_stack.pop(0)
            
            # Clear redo stack when new action is performed
            self.redo_stack.clear()
            self.has_unsaved_changes = True

    def undo(self):
        """Undo last operation"""
        if not self.undo_stack:
            self.status("Nothing to undo")
            return
        
        # Save current state to redo stack
        self.redo_stack.append(compress_image(self.current_image))
        # Restore previous state
        self.current_image = decompress_image(self.undo_stack.pop())
        self.display_image()
        self.speak("Undo applied")

    def redo(self):
        """Redo last undone operation"""
        if not self.redo_stack:
            self.status("Nothing to redo")
            return
        
        # Save current state to undo stack
        self.undo_stack.append(compress_image(self.current_image))
        # Restore next state
        self.current_image = decompress_image(self.redo_stack.pop())
        self.display_image()
        self.speak("Redo applied")

    # ========== Image Filters ==========
    # Each filter saves state before applying for undo functionality

    def apply_grayscale(self):
        """Convert image to grayscale"""
        if not self._check_image(): return
        self.save_image_state()
        
        if self.current_image.ndim == 3:
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.display_image()
        self.speak("Grayscale applied")

    def apply_blur(self):
        """Apply Gaussian blur filter"""
        if not self._check_image(): return
        self.save_image_state()
        
        # 15x15 kernel size for moderate blur
        self.current_image = cv2.GaussianBlur(self.current_image, (15, 15), 0)
        self.display_image()
        self.speak("Blur applied")

    def apply_sharpen(self):
        """Sharpen image using convolution kernel"""
        if not self._check_image(): return
        self.save_image_state()
        
        # Sharpening kernel (enhances edges)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        self.current_image = cv2.filter2D(self.current_image, -1, kernel)
        self.display_image()
        self.speak("Sharpen applied")

    def apply_edge_detection(self):
        """Detect edges using Canny algorithm"""
        if not self._check_image(): return
        self.save_image_state()
        
        # Convert to grayscale first (Canny requires grayscale)
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if self.current_image.ndim == 3 else self.current_image
        # Apply Canny edge detection
        self.current_image = cv2.Canny(gray, 100, 200)
        self.display_image()
        self.speak("Edge detection applied")

    def apply_sepia(self):
        """Apply sepia tone effect for vintage look"""
        if not self._check_image(): return
        self.save_image_state()
        
        img = self.current_image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Sepia transformation matrix
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        self.current_image = cv2.transform(img, kernel)
        self.current_image = np.clip(self.current_image, 0, 255).astype(np.uint8)
        self.display_image()
        self.speak("Sepia applied")

    def apply_invert(self):
        """Invert all colors (negative effect)"""
        if not self._check_image(): return
        self.save_image_state()
        
        self.current_image = cv2.bitwise_not(self.current_image)
        self.display_image()
        self.speak("Invert applied")

    def apply_histogram_equalization(self):
        """Enhance contrast using histogram equalization"""
        if not self._check_image(): return
        self.save_image_state()
        
        img = self.current_image
        if img.ndim == 3:
            # For color images, equalize only the Y (brightness) channel
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            self.current_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            # For grayscale, equalize directly
            self.current_image = cv2.equalizeHist(img)
        self.display_image()
        self.speak("Histogram equalization applied")

    def apply_adaptive_thresholding(self):
        """Apply adaptive threshold for better text/document processing"""
        if not self._check_image(): return
        self.save_image_state()
        
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if self.current_image.ndim == 3 else self.current_image
        # Adaptive threshold adjusts threshold for different areas
        self.current_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.display_image()
        self.speak("Adaptive threshold applied")

    def apply_saturation_default(self):
        """Quick saturation boost (shortcut for slider)"""
        self.saturation_slider.setValue(150)

    # ========== Adjustments ==========
    # These adjust existing image properties using sliders

    def apply_brightness(self, value: int):
        """Adjust brightness by adding/subtracting value from pixels"""
        if not self._check_image(): return
        
        # Store original image on first adjustment
        if not hasattr(self, '_brightness_base'):
            self._brightness_base = self.current_image.copy()
        
        v = clamp(int(value), -100, 100)
        # Add brightness value to each pixel (beta parameter)
        self.current_image = cv2.convertScaleAbs(self._brightness_base, alpha=1, beta=v)
        self.display_image()

    def apply_contrast(self, value: int):
        """Adjust contrast by scaling pixel values"""
        if not self._check_image(): return
        
        # Store original image on first adjustment
        if not hasattr(self, '_contrast_base'):
            self._contrast_base = self.current_image.copy()
        
        v = clamp(int(value), 0, 200)
        # Scale pixels by contrast factor (alpha parameter)
        self.current_image = cv2.convertScaleAbs(self._contrast_base, alpha=v / 100.0, beta=0)
        self.display_image()

    def apply_saturation(self, value: int):
        """Adjust color saturation in HSV color space"""
        if not self._check_image(): return
        
        v = clamp(int(value), 0, 200)
        img = self.current_image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Convert to HSV and adjust S (saturation) channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (v / 100.0), 0, 255)
        hsv = hsv.astype(np.uint8)
        self.current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.display_image()

    def apply_hue(self, value: int):
        """Shift hue (color) in HSV color space"""
        if not self._check_image(): return
        
        v = clamp(int(value), -180, 180)
        img = self.current_image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Convert to HSV and shift H (hue) channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + v) % 180  # Hue is circular (0-180)
        hsv = hsv.astype(np.uint8)
        self.current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.display_image()

    def reset_adjustments(self):
        """Reset all sliders to default values"""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.saturation_slider.setValue(100)
        self.hue_slider.setValue(0)
        
        # Clear cached base images
        if hasattr(self, '_brightness_base'):
            delattr(self, '_brightness_base')
        if hasattr(self, '_contrast_base'):
            delattr(self, '_contrast_base')

    # ========== Transforms ==========

    def rotate_left(self):
        """Rotate image 90 degrees counter-clockwise"""
        if not self._check_image(): return
        self.save_image_state()
        
        self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.display_image()
        self.speak("Rotated left")

    def rotate_right(self):
        """Rotate image 90 degrees clockwise"""
        if not self._check_image(): return
        self.save_image_state()
        
        self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_90_CLOCKWISE)
        self.display_image()
        self.speak("Rotated right")

    def flip_horizontal(self):
        """Flip image horizontally (mirror)"""
        if not self._check_image(): return
        self.save_image_state()
        
        self.current_image = cv2.flip(self.current_image, 1)
        self.display_image()
        self.speak("Flipped horizontally")

    def flip_vertical(self):
        """Flip image vertically (upside down)"""
        if not self._check_image(): return
        self.save_image_state()
        
        self.current_image = cv2.flip(self.current_image, 0)
        self.display_image()
        self.speak("Flipped vertically")

    # ========== Display & Zoom ==========

    def display_image(self):
        """Convert current image to QPixmap and display with zoom level"""
        if self.current_image is None:
            self._reset_image_label()
            return
        
        # Convert NumPy array to Qt format
        qimg = numpy_to_qimage(self.current_image)
        pixmap = QPixmap.fromImage(qimg)
        
        # Apply zoom scaling
        scaled = pixmap.scaled(pixmap.size() * self.zoom_level, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())
        self.image_label.setStyleSheet("")  # Remove placeholder styling
        
        # Update status bar with image info
        h, w = self.current_image.shape[:2]
        unsaved = "*" if self.has_unsaved_changes else ""
        self.status(f"Image: {w}×{h}{unsaved} | Zoom: {int(self.zoom_level * 100)}%")

    def zoom_in(self):
        """Increase zoom level (max 10x)"""
        if not self._check_image(): return
        
        self.zoom_level = min(self.zoom_level * 1.25, 10.0)
        self.display_image()

    def zoom_out(self):
        """Decrease zoom level (min 0.1x)"""
        if not self._check_image(): return
        
        self.zoom_level = max(self.zoom_level / 1.25, 0.1)
        self.display_image()

    def reset_zoom(self):
        """Reset zoom to 100% (actual size)"""
        if not self._check_image(): return
        
        self.zoom_level = 1.0
        self.display_image()

    def fit_to_window(self):
        """Automatically scale image to fit visible area"""
        if not self._check_image(): return
        
        h, w = self.current_image.shape[:2]
        area = self.scroll_area.viewport().size()
        
        # Calculate scale factor to fit window
        if w > 0 and h > 0 and area.width() > 0 and area.height() > 0:
            scale_w = area.width() / w
            scale_h = area.height() / h
            # Use smaller scale to ensure entire image is visible
            self.zoom_level = min(scale_w, scale_h) * 0.95
            self.display_image()

    def reset_image(self):
        """Reset to original loaded image (remove all edits)"""
        if self.original_image is None:
            self.status("⚠️ No original image")
            return
        
        self.save_image_state()  # Allow undoing the reset
        self.current_image = self.original_image.copy()
        self.display_image()
        self.speak("Image reset")

    # ========== Voice Commands ==========

    def handle_command(self, command: str):
        """Process recognized voice command
        
        Uses fuzzy matching to handle similar-sounding commands.
        First checks for parameterized commands (with numbers),
        then checks simple commands using fuzzy matching.
        """
        cmd = command.strip().lower()
        
        # Add to command history
        self.command_history.append(command)
        if len(self.command_history) > self.max_history_size:
            self.command_history.pop(0)
        
        # Update history display (most recent first)
        self.history_list.clear()
        for c in reversed(self.command_history):
            self.history_list.addItem(f"🎤 {c}")
        
        self.status(f"🎤 {command}")
        
        # Map of command keywords to functions
        commands = {
            "grayscale": self.apply_grayscale,
            "blur": self.apply_blur,
            "sharpen": self.apply_sharpen,
            "edge": self.apply_edge_detection,
            "sepia": self.apply_sepia,
            "invert": self.apply_invert,
            "histogram": self.apply_histogram_equalization,
            "adaptive": self.apply_adaptive_thresholding,
            "saturation": self.apply_saturation_default,
            "rotate left": self.rotate_left,
            "rotate right": self.rotate_right,
            "flip horizontal": self.flip_horizontal,
            "flip vertical": self.flip_vertical,
            "zoom in": self.zoom_in,
            "zoom out": self.zoom_out,
            "reset zoom": self.reset_zoom,
            "fit": self.fit_to_window,
            "undo": self.undo,
            "redo": self.redo,
            "reset": self.reset_image,
            "help": self.show_help,
            "exit": self.close,
        }
        
        # Try fuzzy matching for simple commands
        # Allows "grey scale" to match "grayscale", etc.
        best_match = None
        best_score = 0
        for key in commands.keys():
            score = fuzz.ratio(cmd, key)
            if score > 75 and score > best_score:  # 75% similarity threshold
                best_score = score
                best_match = key
        
        if best_match:
            commands[best_match]()
            return
        
        # Check for parameterized commands (e.g., "brightness by 50")
        if m := self.re_brightness.search(cmd):
            self.brightness_slider.setValue(int(m.group(1)))
            return
        
        if m := self.re_contrast.search(cmd):
            self.contrast_slider.setValue(int(m.group(1)))
            return
        
        if m := self.re_saturation.search(cmd):
            self.saturation_slider.setValue(int(m.group(1)))
            return
        
        if m := self.re_hue.search(cmd):
            self.hue_slider.setValue(int(m.group(1)))
            return
        
        # Command not recognized
        self.status("⚠️ Command not recognized")

    # ========== Utilities ==========

    def _check_image(self) -> bool:
        """Check if an image is loaded before applying operations"""
        if self.current_image is None:
            self.status("⚠️ No image loaded")
            return False
        return True

    def status(self, msg: str):
        """Update status bar message"""
        self.status_label.setText(msg)

    def show_help(self):
        """Show help dialog"""
        dialog = HelpDialog(self)
        dialog.exec()

    # ========== Events ==========

    def eventFilter(self, obj, event):
        """Event filter for Ctrl+Wheel zoom on image"""
        if obj is self.image_label and isinstance(event, QWheelEvent):
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                if self.current_image is not None:
                    # Wheel up = zoom in, wheel down = zoom out
                    if event.angleDelta().y() > 0:
                        self.zoom_in()
                    else:
                        self.zoom_out()
                    return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        """Handle window close - check for unsaved changes"""
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self, 'Unsaved Changes',
                'You have unsaved changes. Do you want to exit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # Clean up threads before closing
        self._cleanup()
        event.accept()

    def _cleanup(self):
        """Stop all background threads gracefully"""
        try:
            if hasattr(self, "speech_thread"):
                self.speech_thread.stop_listening()
                self.speech_thread.wait(2000)
        except: 
            pass
        
        try:
            if hasattr(self, "tts_thread"):
                self.tts_thread.stop()
                self.tts_thread.wait(2000)
        except: 
            pass


# ========== Main Entry Point ==========

if __name__ == "__main__":
    # Create application instance
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    app.setApplicationName("Photo Editor Pro")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())