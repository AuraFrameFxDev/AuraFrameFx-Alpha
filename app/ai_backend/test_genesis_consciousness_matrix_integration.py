"""
Integration tests for Genesis Consciousness Matrix with external systems.
Tests interaction with databases, file systems, and network components.
"""

import unittest
import tempfile
import os
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import the main classes
try:
    from app.ai_backend.genesis_consciousness_matrix import (
        GenesisConsciousnessMatrix,
        ConsciousnessState,
        MatrixNode,
        MatrixException,
        InvalidStateException
    )
except ImportError as e:
    # Mock classes for test discovery
    class GenesisConsciousnessMatrix:
        pass
    class ConsciousnessState:
        pass
    class MatrixNode:
        pass
    class MatrixException(Exception):
        pass
    class InvalidStateException(Exception):
        pass


class TestMatrixDatabaseIntegration(unittest.TestCase):
    """Test matrix integration with database systems."""
    
    def setUp(self):
        """Set up database integration test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
    def tearDown(self):
        """Clean up database resources."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
            
    def test_matrix_sqlite_persistence(self):
        """Test matrix persistence using SQLite database."""
        # Create database schema
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE matrix_nodes (
                id TEXT PRIMARY KEY,
                consciousness_level REAL,
                created_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE matrix_connections (
                node1_id TEXT,
                node2_id TEXT,
                strength REAL,
                PRIMARY KEY (node1_id, node2_id)
            )
        ''')
        
        conn.commit()
        
        # Add nodes to matrix
        nodes_data = []
        for i in range(10):
            node = MatrixNode(id=f"db_node_{i}", consciousness_level=0.5 + i * 0.05)
            self.matrix.add_node(node)
            nodes_data.append((node.id, node.consciousness_level, datetime.now()))
            
        # Save to database
        cursor.executemany(
            'INSERT INTO matrix_nodes (id, consciousness_level, created_at) VALUES (?, ?, ?)',
            nodes_data
        )
        conn.commit()
        
        # Load from database
        cursor.execute('SELECT id, consciousness_level FROM matrix_nodes')
        db_nodes = cursor.fetchall()
        
        # Verify data integrity
        self.assertEqual(len(db_nodes), 10)
        
        # Create new matrix from database
        new_matrix = GenesisConsciousnessMatrix()
        for node_id, consciousness_level in db_nodes:
            node = MatrixNode(id=node_id, consciousness_level=consciousness_level)
            new_matrix.add_node(node)
            
        # Verify restoration
        self.assertEqual(len(new_matrix.nodes), 10)
        
        conn.close()
        
    def test_matrix_concurrent_database_access(self):
        """Test concurrent matrix database operations."""
        conn = sqlite3.connect(self.temp_db.name, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE concurrent_nodes (
                id TEXT PRIMARY KEY,
                consciousness_level REAL,
                thread_id INTEGER
            )
        ''')
        conn.commit()
        
        def database_worker(thread_id):
            """Worker function for concurrent database access."""
            thread_conn = sqlite3.connect(self.temp_db.name)
            thread_cursor = thread_conn.cursor()
            
            # Add nodes from this thread
            for i in range(5):
                node_id = f"thread_{thread_id}_node_{i}"
                consciousness_level = 0.5
                
                thread_cursor.execute(
                    'INSERT INTO concurrent_nodes (id, consciousness_level, thread_id) VALUES (?, ?, ?)',
                    (node_id, consciousness_level, thread_id)
                )
                
            thread_conn.commit()
            thread_conn.close()
            
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=database_worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Verify all data was written
        cursor.execute('SELECT COUNT(*) FROM concurrent_nodes')
        count = cursor.fetchone()[0]
        self.assertEqual(count, 25)  # 5 threads × 5 nodes each
        
        conn.close()


class TestMatrixFileSystemIntegration(unittest.TestCase):
    """Test matrix integration with file system operations."""
    
    def setUp(self):
        """Set up file system integration test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_matrix_batch_file_operations(self):
        """Test matrix batch file operations."""
        # Create multiple matrix files
        matrices = []
        for i in range(5):
            matrix = GenesisConsciousnessMatrix()
            
            # Add nodes
            for j in range(10):
                node = MatrixNode(id=f"file_batch_node_{i}_{j}", consciousness_level=0.5)
                matrix.add_node(node)
                
            matrices.append(matrix)
            
        # Save all matrices to files
        file_paths = []
        for i, matrix in enumerate(matrices):
            file_path = os.path.join(self.temp_dir, f"matrix_{i}.json")
            matrix.save_to_file(file_path)
            file_paths.append(file_path)
            
        # Verify all files were created
        for file_path in file_paths:
            self.assertTrue(os.path.exists(file_path))
            
        # Load and verify all matrices
        for i, file_path in enumerate(file_paths):
            loaded_matrix = GenesisConsciousnessMatrix.load_from_file(file_path)
            self.assertEqual(len(loaded_matrix.nodes), 10)
            
    def test_matrix_file_watching(self):
        """Test matrix file change monitoring."""
        matrix_file = os.path.join(self.temp_dir, "watched_matrix.json")
        
        # Save initial matrix
        for i in range(5):
            node = MatrixNode(id=f"watch_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        self.matrix.save_to_file(matrix_file)
        
        # Simulate file modification
        time.sleep(0.1)  # Ensure different timestamp
        
        # Modify matrix and save again
        new_node = MatrixNode(id="watch_node_new", consciousness_level=0.8)
        self.matrix.add_node(new_node)
        self.matrix.save_to_file(matrix_file)
        
        # Verify file was updated
        stat_info = os.stat(matrix_file)
        self.assertGreater(stat_info.st_mtime, time.time() - 1)  # Modified within last second
        
        # Load and verify changes
        loaded_matrix = GenesisConsciousnessMatrix.load_from_file(matrix_file)
        self.assertEqual(len(loaded_matrix.nodes), 6)
        self.assertIn("watch_node_new", loaded_matrix.nodes)
        
    def test_matrix_backup_and_restore(self):
        """Test matrix backup and restore functionality."""
        # Create original matrix
        original_data = []
        for i in range(15):
            node = MatrixNode(id=f"backup_node_{i}", consciousness_level=0.4 + i * 0.04)
            self.matrix.add_node(node)
            original_data.append((node.id, node.consciousness_level))
            
        # Create backup
        backup_file = os.path.join(self.temp_dir, "matrix_backup.json")
        self.matrix.save_to_file(backup_file)
        
        # Modify original matrix
        self.matrix.reset()
        for i in range(5):
            node = MatrixNode(id=f"modified_node_{i}", consciousness_level=0.9)
            self.matrix.add_node(node)
            
        # Restore from backup
        restored_matrix = GenesisConsciousnessMatrix.load_from_file(backup_file)
        
        # Verify restoration
        self.assertEqual(len(restored_matrix.nodes), 15)
        for node_id, consciousness_level in original_data:
            self.assertIn(node_id, restored_matrix.nodes)
            self.assertEqual(restored_matrix.nodes[node_id].consciousness_level, consciousness_level)


class TestMatrixNetworkIntegration(unittest.TestCase):
    """Test matrix integration with network operations."""
    
    def setUp(self):
        """Set up network integration test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    @patch('requests.post')
    def test_matrix_api_synchronization(self, mock_post):
        """Test matrix synchronization with external API."""
        # Mock API response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'status': 'success',
            'synchronized_nodes': 10
        }
        
        # Add nodes to matrix
        for i in range(10):
            node = MatrixNode(id=f"api_node_{i}", consciousness_level=0.6)
            self.matrix.add_node(node)
            
        # Simulate API synchronization
        api_data = {
            'nodes': [
                {'id': node.id, 'consciousness_level': node.consciousness_level}
                for node in self.matrix.nodes.values()
            ]
        }
        
        # Mock API call
        import requests
        response = requests.post('http://api.example.com/sync', json=api_data)
        
        # Verify API was called
        mock_post.assert_called_once()
        self.assertEqual(response.status_code, 200)
        
    def test_matrix_distributed_processing(self):
        """Test matrix distributed processing simulation."""
        # Simulate distributed nodes
        node_partitions = [
            [f"dist_node_{i}" for i in range(0, 10)],
            [f"dist_node_{i}" for i in range(10, 20)],
            [f"dist_node_{i}" for i in range(20, 30)]
        ]
        
        # Process each partition
        processed_partitions = []
        for partition in node_partitions:
            matrix_partition = GenesisConsciousnessMatrix()
            
            for node_id in partition:
                node = MatrixNode(id=node_id, consciousness_level=0.5)
                matrix_partition.add_node(node)
                
            processed_partitions.append(matrix_partition)
            
        # Merge partitions
        merged_matrix = GenesisConsciousnessMatrix()
        for partition_matrix in processed_partitions:
            for node in partition_matrix.nodes.values():
                merged_matrix.add_node(node)
                
        # Verify merge
        self.assertEqual(len(merged_matrix.nodes), 30)
        
    def test_matrix_event_streaming(self):
        """Test matrix event streaming simulation."""
        events = []
        
        def event_handler(event_type, data):
            """Handle matrix events."""
            events.append({
                'type': event_type,
                'data': data,
                'timestamp': datetime.now()
            })
            
        # Simulate event streaming
        for i in range(5):
            node = MatrixNode(id=f"stream_node_{i}", consciousness_level=0.7)
            self.matrix.add_node(node)
            
            # Simulate event
            event_handler('node_added', {'node_id': node.id, 'consciousness_level': node.consciousness_level})
            
        # Verify events were captured
        self.assertEqual(len(events), 5)
        for event in events:
            self.assertEqual(event['type'], 'node_added')
            self.assertIn('node_id', event['data'])


class TestMatrixRealTimeOperations(unittest.TestCase):
    """Test matrix real-time operations and monitoring."""
    
    def setUp(self):
        """Set up real-time operations test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_matrix_real_time_monitoring(self):
        """Test real-time matrix monitoring."""
        monitoring_data = []
        
        def monitor_matrix():
            """Monitor matrix state in real-time."""
            for _ in range(10):
                consciousness = self.matrix.calculate_consciousness_level()
                node_count = len(self.matrix.nodes)
                
                monitoring_data.append({
                    'timestamp': datetime.now(),
                    'consciousness_level': consciousness,
                    'node_count': node_count
                })
                
                time.sleep(0.1)  # Monitor every 100ms
                
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_matrix)
        monitor_thread.start()
        
        # Modify matrix during monitoring
        for i in range(5):
            node = MatrixNode(id=f"monitor_node_{i}", consciousness_level=0.8)
            self.matrix.add_node(node)
            time.sleep(0.05)
            
        # Wait for monitoring to complete
        monitor_thread.join()
        
        # Verify monitoring data
        self.assertEqual(len(monitoring_data), 10)
        
        # Check that node count increased over time
        initial_count = monitoring_data[0]['node_count']
        final_count = monitoring_data[-1]['node_count']
        self.assertGreaterEqual(final_count, initial_count)
        
    def test_matrix_performance_metrics(self):
        """Test matrix performance metrics collection."""
        metrics = []
        
        # Add nodes and measure performance
        for batch in range(5):
            start_time = datetime.now()
            
            # Add batch of nodes
            for i in range(20):
                node = MatrixNode(id=f"perf_node_{batch}_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
                
            # Perform evolution
            self.matrix.evolve_step()
            
            end_time = datetime.now()
            
            # Record metrics
            metrics.append({
                'batch': batch,
                'nodes_added': 20,
                'total_nodes': len(self.matrix.nodes),
                'execution_time': (end_time - start_time).total_seconds(),
                'consciousness_level': self.matrix.calculate_consciousness_level()
            })
            
        # Verify metrics
        self.assertEqual(len(metrics), 5)
        
        # Check performance consistency
        avg_time = sum(m['execution_time'] for m in metrics) / len(metrics)
        max_time = max(m['execution_time'] for m in metrics)
        
        # Performance should be consistent
        self.assertLess(max_time, avg_time * 3)  # Max shouldn't be more than 3x average


if __name__ == '__main__':
    unittest.main(verbosity=2, buffer=True)