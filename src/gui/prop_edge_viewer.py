"""GUI application for viewing and updating prop betting edges."""

import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QComboBox,
    QSpinBox,
    QHeaderView,
    QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

from src.models.find_prop_edges import PropEdgeFinder

class PropEdgeViewer(QMainWindow):
    """Main window for prop edge viewing application."""
    
    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        
        # Initialize edge finder
        api_key = os.getenv('ODDS_API_KEY')
        if not api_key:
            raise ValueError("ODDS_API_KEY environment variable not set")
        self.edge_finder = PropEdgeFinder(api_key)
        
        self.setWindowTitle("Prop Edge Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create control panel
        control_layout = QHBoxLayout()
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh Odds")
        self.refresh_button.clicked.connect(self.refresh_odds)
        control_layout.addWidget(self.refresh_button)
        
        # Retrain model button
        self.retrain_button = QPushButton("Retrain Models")
        self.retrain_button.clicked.connect(self.retrain_models)
        control_layout.addWidget(self.retrain_button)
        
        # Prop type filter
        prop_label = QLabel("Prop Type:")
        control_layout.addWidget(prop_label)
        
        self.prop_filter = QComboBox()
        self.prop_filter.addItems(['All', 'points', 'rebounds', 'assists', 'threes'])
        self.prop_filter.currentTextChanged.connect(self.apply_filters)
        control_layout.addWidget(self.prop_filter)
        
        # Min edge filter
        edge_label = QLabel("Min Edge %:")
        control_layout.addWidget(edge_label)
        
        self.edge_filter = QSpinBox()
        self.edge_filter.setRange(0, 100)
        self.edge_filter.setValue(5)
        self.edge_filter.valueChanged.connect(self.apply_filters)
        control_layout.addWidget(self.edge_filter)
        
        # Add auto-refresh toggle
        self.auto_refresh = QPushButton("Auto-Refresh: Off")
        self.auto_refresh.setCheckable(True)
        self.auto_refresh.clicked.connect(self.toggle_auto_refresh)
        control_layout.addWidget(self.auto_refresh)
        
        # Status label
        self.status_label = QLabel()
        control_layout.addWidget(self.status_label)
        
        # Add stretch to push controls to the left
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            'Player',
            'Prop Type',
            'Line',
            'Prediction',
            'Best Odds',
            'Book',
            'Edge %',
            'Bet Type'
        ])
        
        # Set table properties
        header = self.table.horizontalHeader()
        for i in range(8):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        
        layout.addWidget(self.table)
        
        # Set up auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_odds)
        
        # Initial data load
        self.refresh_odds()
    
    def toggle_auto_refresh(self) -> None:
        """Toggle auto-refresh timer."""
        if self.auto_refresh.isChecked():
            self.auto_refresh.setText("Auto-Refresh: On")
            self.refresh_timer.start(60000)  # Refresh every minute
        else:
            self.auto_refresh.setText("Auto-Refresh: Off")
            self.refresh_timer.stop()
    
    def refresh_odds(self) -> None:
        """Refresh odds data and update table."""
        try:
            self.status_label.setText("Refreshing odds...")
            self.refresh_button.setEnabled(False)
            QApplication.processEvents()
            
            # Get edges
            edges_df = self.edge_finder.find_edges()
            
            if edges_df.empty:
                self.status_label.setText("No edges found")
                self.table.setRowCount(0)
                return
            
            # Apply filters
            prop_filter = self.prop_filter.currentText()
            if prop_filter != 'All':
                edges_df = edges_df[edges_df['prop_type'] == prop_filter]
            
            min_edge = self.edge_filter.value()
            edges_df = edges_df[edges_df['edge'].abs() >= min_edge]
            
            # Sort by absolute edge value
            edges_df = edges_df.iloc[edges_df['edge'].abs().argsort()[::-1]]
            
            # Update table
            self.table.setRowCount(len(edges_df))
            
            for i, (_, row) in enumerate(edges_df.iterrows()):
                self.table.setItem(i, 0, QTableWidgetItem(row['player']))
                self.table.setItem(i, 1, QTableWidgetItem(row['prop_type']))
                self.table.setItem(i, 2, QTableWidgetItem(f"{row['line']:.1f}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{row['prediction']:.1f}"))
                self.table.setItem(i, 4, QTableWidgetItem(f"{row['odds']}"))
                self.table.setItem(i, 5, QTableWidgetItem(row['book']))
                
                # Color edge cell based on value
                edge_item = QTableWidgetItem(f"{row['edge']:.1f}%")
                if row['edge'] > 0:
                    edge_item.setBackground(QColor(200, 255, 200))  # Light green
                else:
                    edge_item.setBackground(QColor(255, 200, 200))  # Light red
                self.table.setItem(i, 6, edge_item)
                
                self.table.setItem(i, 7, QTableWidgetItem(row['bet_type']))
            
            self.status_label.setText(
                f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to refresh odds: {str(e)}"
            )
            self.status_label.setText("Error refreshing odds")
        
        finally:
            self.refresh_button.setEnabled(True)
    
    def retrain_models(self) -> None:
        """Retrain all models."""
        try:
            self.status_label.setText("Retraining models...")
            self.retrain_button.setEnabled(False)
            QApplication.processEvents()
            
            # Create new edge finder with force_retrain
            api_key = os.getenv('ODDS_API_KEY')
            self.edge_finder = PropEdgeFinder(api_key, force_retrain=True)
            
            # Refresh odds with new models
            self.refresh_odds()
            
            self.status_label.setText("Models retrained successfully")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to retrain models: {str(e)}"
            )
            self.status_label.setText("Error retraining models")
        
        finally:
            self.retrain_button.setEnabled(True)
    
    def apply_filters(self) -> None:
        """Apply prop type and edge filters."""
        self.refresh_odds()

def main() -> int:
    """Run the prop edge viewer application.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        app = QApplication(sys.argv)
        window = PropEdgeViewer()
        window.show()
        return app.exec()
        
    except Exception as e:
        print(f"Application failed to start: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
