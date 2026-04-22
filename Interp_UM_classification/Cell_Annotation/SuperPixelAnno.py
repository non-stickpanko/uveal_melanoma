from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDir, Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout,
                             QLabel, QPushButton, QSizePolicy, QWidget,
                             QVBoxLayout, QGraphicsView, QGraphicsScene)

import json
import numpy as np
from util import *
from copy import deepcopy
import os
from skimage.io import imread


def numpy_to_qimage(arr):
    """Convert numpy array to QImage.
    
    Args:
        arr: Numpy array in RGB or grayscale format
        
    Returns:
        QImage: Qt image object
    """
    if arr is None:
        return QImage()
    
    # Ensure uint8 format
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    
    # Ensure C-contiguous array
    arr = np.ascontiguousarray(arr)
    
    # Handle 3D arrays (RGB images)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        h, w, _ = arr.shape
        # Ensure RGB format (3 channels)
        if arr.shape[2] == 3:
            rgb_array = arr
        else:
            rgb_array = arr[:, :, :3]
        bytes_per_line = 3 * w
        qimage = QImage(rgb_array.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        return qimage.copy()  # Return a copy to avoid memory issues
    # Handle 2D arrays (grayscale)
    elif arr.ndim == 2:
        h, w = arr.shape
        bytes_per_line = w
        qimage = QImage(arr.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)
        return qimage.copy()  # Return a copy to avoid memory issues
    else:
        return QImage()


class PhotoViewer(QtWidgets.QGraphicsView):
    """Image viewer with zoom and pan capabilities."""
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        """Initialize the photo viewer.
        
        Args:
            parent: Parent widget
        """
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._isPanning = False
        self._mousePressedRight = False
    def hasPhoto(self):
        return not self._empty

    def fitInView(self):
        """Fit image to view."""
        rect_image = QtCore.QRectF(self._photo.pixmap().rect())
        rect = QtCore.QRectF(self.rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewport = self.viewport()
                if viewport:
                    viewrect = viewport.rect()
                    scenerect = self.transform().mapRect(rect_image)
                    factor = min(viewrect.width() / scenerect.width(),
                                 viewrect.height() / scenerect.height())
                    self.scale(factor, factor)
            self._zoom = 0
    def setPhoto(self, pixmap=None):
        """Set image to display.
        
        Args:
            pixmap: QPixmap to display
        """
        if pixmap:
            self.pixmap = pixmap
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()




class Window(QtWidgets.QWidget):
    """Main application window for super-pixel annotation."""
    
    def __init__(self):
        """Initialize the application window."""
        super(Window, self).__init__()
        
        # Initialize image data attributes
        self.image = None
        self.SLIC = None
        self.SLIC_visualization = None
        self.SLIC_visualization_ori = None
        self.image_path = None
        self.image_name = None

        self.viewer1 = PhotoViewer(self)
        self.viewer2 = PhotoViewer(self)

        self.viewer1.setAcceptDrops(True)
        self.viewer1.setMouseTracking(True)
        self.viewer2.setAcceptDrops(True)
        self.viewer2.setMouseTracking(True)
        
        vp1 = self.viewer1.viewport()
        vp2 = self.viewer2.viewport()
        if vp1:
            vp1.installEventFilter(self)
        if vp2:
            vp2.installEventFilter(self)

        browse_ImagePathButton = self.createButton("&Browse...", self.browse_ImagePath)
        LoadImageButton = self.createButton("&Load Image", self.loadImage)
        FinishButton = self.createButton("&Finish and save...", self.finish)
        GoodButton = self.createButton("&Good", self.Good)
        BadButton = self.createButton("&Bad", self.Bad)

        self.directoryComboBox_ImagePath = self.createComboBox(QDir.currentPath())
        directoryLabel_ImagePath = QLabel("Image Path:")

        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        HBlayout0 = QtWidgets.QHBoxLayout()
        HBlayout0.setAlignment(Qt.AlignLeft)  # type: ignore
        HBlayout0.addWidget(self.viewer1)
        HBlayout0.addWidget(self.viewer2)
        VBlayout.addLayout(HBlayout0)

        VBlayout1 = QtWidgets.QVBoxLayout()
        HBlayout2 = QtWidgets.QHBoxLayout()
        HBlayout2.setAlignment(Qt.AlignLeft)  # type: ignore
        HBlayout2.addWidget(directoryLabel_ImagePath)
        HBlayout2.addWidget(self.directoryComboBox_ImagePath)
        HBlayout2.addWidget(browse_ImagePathButton)
        VBlayout1.addLayout(HBlayout2)

        HBlayout6 = QtWidgets.QHBoxLayout()
        HBlayout6.setAlignment(Qt.AlignRight)  # type: ignore
        HBlayout6.addWidget(GoodButton)
        HBlayout6.addWidget(BadButton)
        HBlayout6.addWidget(LoadImageButton)
        HBlayout6.addWidget(FinishButton)
        VBlayout1.addLayout(HBlayout6)
        VBlayout.addLayout(VBlayout1)

        self.anno_all = {'good': [], 'bad': []}
        self.anno_temp = []

    def loadImage(self):
        """Load and display image with SLIC visualization."""
        if self.image is None or self.SLIC is None:
            print("Error: Image or SLIC data not loaded")
            return
        
        if self.image.ndim < 2:
            print("Error: Image must be at least 2D")
            return
        
        if self.SLIC is None or self.SLIC.ndim != 2:
            print("Error: SLIC data must be 2D")
            return
        
        try:
            self.InitializationPreparation()
            if self.image is not None:
                qim = numpy_to_qimage(self.image.astype(np.uint8) if self.image.dtype != np.uint8 else self.image)
                self.viewer1.setPhoto(QPixmap(qim))
            
            if self.SLIC_visualization is not None:
                qim_state = numpy_to_qimage(self.SLIC_visualization.astype(np.uint8))
                self.viewer2.setPhoto(QPixmap(qim_state))
            
            self.anno_all = {'good': [], 'bad': []}
            self.anno_temp = []
        except Exception as e:
            print(f"Error loading image: {e}")

    def done_color(self, labels, r, g, b):
        """Update SLIC visualization with new colors.
        
        Args:
            labels: List of SLIC labels to recolor
            r: Red value (0-255)
            g: Green value (0-255)
            b: Blue value (0-255)
        """
        if self.SLIC is None or self.SLIC_visualization is None:
            return
        
        for label in labels:
            if np.any(self.SLIC == label):
                self.SLIC_visualization[self.SLIC == label, 0] = r
                self.SLIC_visualization[self.SLIC == label, 1] = g
                self.SLIC_visualization[self.SLIC == label, 2] = b
        
        qim_state = numpy_to_qimage(self.SLIC_visualization.astype(np.uint8))
        self.viewer2.setPhoto(QPixmap(qim_state))
        hbar = self.viewer2.horizontalScrollBar()
        vbar = self.viewer2.verticalScrollBar()
        if hbar:
            hbar.setValue(0)
        if vbar:
            vbar.setValue(0)

    def Good(self):
        """Mark current annotation as good."""
        if not self.anno_temp:
            print("No pixels selected")
            return
        
        self.anno_all['good'].append(list(set(self.anno_temp)))
        self.done_color(self.anno_temp, 255, 255, 0)
        self.anno_temp = []

    def Bad(self):
        """Mark current annotation as bad."""
        if not self.anno_temp:
            print("No pixels selected")
            return
        
        self.anno_all['bad'].append(list(set(self.anno_temp)))
        self.done_color(self.anno_temp, 0, 0, 255)
        self.anno_temp = []

    def browse(self, phase=None):
        """Browse and load image file.
        
        Args:
            phase: Phase identifier (currently only 'ImagePath' is supported)
        """
        if phase != 'ImagePath':
            return
        
        default_dir = self.image_path if self.image_path else QDir.currentPath()
        directory = QFileDialog.getOpenFileName(self, "Find Files", default_dir)
        
        if not directory or not directory[0]:
            return
        
        path = directory[0]
        if self.directoryComboBox_ImagePath.findText(path) == -1:
            self.directoryComboBox_ImagePath.addItem(path)
        self.directoryComboBox_ImagePath.setCurrentIndex(
            self.directoryComboBox_ImagePath.findText(path))
        
        self.image_path = os.path.dirname(path)
        self.image_name = os.path.splitext(os.path.basename(path))[0]
        
        try:
            self.image = imread(path)
            slic_path = os.path.join(self.image_path, self.image_name + '.npy')
            self.SLIC = np.load(slic_path)
            self.loadImage()
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except Exception as e:
            print(f"Error loading image or SLIC data: {e}")

    def browse_ImagePath(self):
        self.browse('ImagePath')

    def InitializationPreparation(self):
        """Create SLIC visualization from image and labels."""
        if self.image is None or self.SLIC is None:
            return
        
        self.SLIC_visualization = np.zeros_like(self.image, dtype=np.float32)
        for label in np.unique(self.SLIC):
            mask = self.SLIC == label
            num_channels = min(3, self.image.shape[2] if self.image.ndim >= 3 else 1)
            for j in range(num_channels):
                if self.image.ndim >= 3:
                    self.SLIC_visualization[mask, j] = np.mean(self.image[mask, j])
                else:
                    self.SLIC_visualization[mask, j] = np.mean(self.image[mask])
        
        self.SLIC_visualization_ori = self.SLIC_visualization.copy()

    def createButton(self, text, member):
        """Create a push button.
        
        Args:
            text: Button label
            member: Function to call when clicked
            
        Returns:
            QPushButton: Created button
        """
        button = QPushButton(text)
        button.clicked.connect(member)
        return button
    
    def createComboBox(self, text=""):
        """Create a combo box.
        
        Args:
            text: Initial text/item
            
        Returns:
            QComboBox: Created combo box
        """
        comboBox = QComboBox()
        comboBox.setEditable(True)
        comboBox.addItem(text)
        comboBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return comboBox

    def eventFilter(self, obj, event):
        """Handle mouse click events for label selection.
        
        Args:
            obj: Object that triggered the event
            event: Event information
            
        Returns:
            bool: True if event was handled
        """
        if event.type() != QEvent.MouseButtonPress:  # type: ignore
            return False
        
        if event.button() != Qt.RightButton:  # type: ignore
            return False
        
        if event.globalX() < self.viewer2.x():
            return False
        
        if self.SLIC is None or self.SLIC_visualization is None:
            return False
        
        try:
            scale_factor = self.viewer2.transform().m11()
            if scale_factor == 0:
                return False
            
            hbar = self.viewer2.horizontalScrollBar()
            vbar = self.viewer2.verticalScrollBar()
            if not hbar or not vbar:
                return False
            
            canvas_x = hbar.value() + event.x()
            canvas_y = vbar.value() + event.y()
            
            x = int(canvas_y // scale_factor)
            y = int(canvas_x // scale_factor)
            
            # Bounds checking
            if x < 0 or x >= self.SLIC.shape[0] or y < 0 or y >= self.SLIC.shape[1]:
                return False
            
            label = int(self.SLIC[x, y])
            
            if label not in self.anno_temp:
                self.done_color([label], 0, 255, 0)
                self.anno_temp.append(label)
            else:
                if self.SLIC_visualization_ori is not None:
                    r = int(self.SLIC_visualization_ori[x, y, 0])
                    g = int(self.SLIC_visualization_ori[x, y, 1])
                    b = int(self.SLIC_visualization_ori[x, y, 2])
                    self.done_color([label], r, g, b)
                while label in self.anno_temp:
                    self.anno_temp.remove(label)
            
            if hbar:
                hbar.setValue(0)
            if vbar:
                vbar.setValue(0)
        except (IndexError, ValueError, ZeroDivisionError) as e:
            print(f"Error processing click: {e}")
            return False
        
        return False
    
    def keyPressEvent(self, e):
        """Handle keyboard shortcuts.
        
        Args:
            e: Key event
        """
        if e.key() == Qt.Key_Q:  # type: ignore
            self.Good()
        elif e.key() == Qt.Key_E:  # type: ignore
            self.Bad()

    def finish(self):
        """Save annotations to file and reset state."""
        if not self.image_path or not self.image_name:
            print("Error: No image loaded. Please load an image first.")
            return
        
        try:
            anno_file = os.path.join(self.image_path, self.image_name + '.anno')
            with open(anno_file, 'w') as f:
                json.dump(self.anno_all, f, indent=4)
            print(f"Annotations saved to {anno_file}")
        except IOError as e:
            print(f"Error saving annotations: {e}")
            return
        
        # Reset state
        self.anno_all = {'good': [], 'bad': []}
        self.anno_temp = []
        self.SLIC_visualization = None
        self.image = None
        self.SLIC = None



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setGeometry(200, 100, 1600, 1000)
    window.show()
    sys.exit(app.exec_())