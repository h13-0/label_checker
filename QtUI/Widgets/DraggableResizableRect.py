from PyQt6.QtWidgets import QGraphicsRectItem
from PyQt6.QtGui import QPen, QColor, QBrush, QCursor
from PyQt6.QtCore import Qt, QRectF

class DraggableResizableRect(QGraphicsRectItem):
    def __init__(self, 
            x:int, 
            y:int, 
            width:int, 
            height:int, 
            fill_color:QColor,
            edge_color:QColor,
            edge_size:int,
            parent=None
        ):
        super().__init__(x, y, width, height, parent)
        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.pen = QPen(edge_color)
        self.pen.setWidth(edge_size)
        self.brush = QBrush(edge_color)
        self.setPen(self.pen)
        self.setBrush(self.brush)
        self.edge_size = edge_size
        self.current_cursor = Qt.CursorShape.ArrowCursor
        self.resizing = False
        self.dragging = False


    def mousePressEvent(self, event):
        """根据鼠标点击事件判定操作类型, 并修改当前控件状态"""
        self._current_edge = self._detect_edge(event.pos())
        if self._current_edge:
            self.resizing = True
            self.dragging = False
            self.origin = event.pos()
            self.origRect = QRectF(self.rect())
        else:
            self.resizing = False
            self.dragging = True
            
        super().mousePressEvent(event)


    def mouseMoveEvent(self, event):
        """根据当前控件状态执行不同逻辑"""
        if self.dragging:
            ## 处理拖动
            super().mouseMoveEvent(event)
        elif self.resizing:
            ## 处理resize事件
            self.setCursor(self._cursor_for_edge(self._current_edge))
            self._resize_rect(event.pos())
        else:
            pass


    def mouseReleaseEvent(self, event):
        """当鼠标松开左键时, 取消拖动和修改大小状态"""
        self.dragging = False
        self.resizing = False
        # 恢复鼠标状态
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().mouseReleaseEvent(event)


    def _detect_edge(self, pos):
        """检测鼠标是否在边缘或角落"""
        rect = self.rect()
        if pos.x() <= rect.left() + self.edge_size:
            if pos.y() <= rect.top() + self.edge_size:
                return 'top_left'
            elif pos.y() >= rect.bottom() - self.edge_size:
                return 'bottom_left'
            return 'left'
        elif pos.x() >= rect.right() - self.edge_size:
            if pos.y() <= rect.top() + self.edge_size:
                return 'top_right'
            elif pos.y() >= rect.bottom() - self.edge_size:
                return 'bottom_right'
            return 'right'
        elif pos.y() <= rect.top() + self.edge_size:
            return 'top'
        elif pos.y() >= rect.bottom() - self.edge_size:
            return 'bottom'
        return None


    def _cursor_for_edge(self, edge):
        """获取当前位置对应的鼠标指针形状"""
        cursors = {
            'top_left': Qt.CursorShape.SizeFDiagCursor,
            'bottom_right': Qt.CursorShape.SizeFDiagCursor,
            'top_right': Qt.CursorShape.SizeBDiagCursor,
            'bottom_left': Qt.CursorShape.SizeBDiagCursor,
            'top': Qt.CursorShape.SizeVerCursor,
            'bottom': Qt.CursorShape.SizeVerCursor,
            'left': Qt.CursorShape.SizeHorCursor,
            'right': Qt.CursorShape.SizeHorCursor,
        }
        return QCursor(cursors.get(edge, Qt.CursorShape.ArrowCursor))


    def hoverLeaveEvent(self, event):
        """鼠标离开事件"""
        super().hoverLeaveEvent(event)


    def itemChange(self, change, value):
        #if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionChange and self.scene():
            # 这里可以添加代码以处理或限制位置变化
        #    print(change)
        #    print(value)
        return super().itemChange(change, value)


    def _resize_rect(self, newPos):
        rect = QRectF(self.origRect)
        if 'left' in self._current_edge:
            rect.setLeft(newPos.x())
        if 'right' in self._current_edge:
            rect.setRight(newPos.x())
        if 'top' in self._current_edge:
            rect.setTop(newPos.y())
        if 'bottom' in self._current_edge:
            rect.setBottom(newPos.y())
        self.setRect(rect)