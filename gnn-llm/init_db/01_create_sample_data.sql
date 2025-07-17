-- Create sample tables and data for testing

-- Sample documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Sample products table
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    price DECIMAL(10,2),
    tags TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample documents
INSERT INTO documents (title, content, category, metadata) VALUES
('Machine Learning Basics', 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.', 'AI', '{"difficulty": "beginner"}'),
('Graph Neural Networks', 'Graph Neural Networks (GNNs) are designed to work with graph-structured data. They can capture relationships between nodes.', 'AI', '{"difficulty": "advanced"}'),
('Database Design', 'Good database design follows normalization principles to reduce redundancy and improve data integrity.', 'Database', '{"difficulty": "beginner"}');

-- Insert sample products  
INSERT INTO products (name, description, category, price, tags) VALUES
('Wireless Headphones', 'High-quality wireless headphones with noise cancellation.', 'Electronics', 199.99, ARRAY['audio', 'wireless']),
('Smart Watch', 'Fitness tracking smartwatch with heart rate monitor.', 'Electronics', 299.99, ARRAY['fitness', 'smartwatch']),
('Laptop Stand', 'Ergonomic aluminum laptop stand.', 'Accessories', 49.99, ARRAY['ergonomic', 'laptop']);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO graphuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO graphuser;