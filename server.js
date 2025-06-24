const express = require('express');
const { Pool } = require('pg');
const cors = require('cors');
const app = express();
const port = process.env.PORT || 3001;

// Enable CORS
app.use(cors());
app.use(express.json());

// Create a new pool using the connection string from environment variable
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: {
    rejectUnauthorized: false
  }
});

// API endpoint for hitters
app.get('/api/hitters', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM hitters');
    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching hitters:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// API endpoint for pitchers
app.get('/api/pitchers', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM pitchers');
    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching pitchers:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(port, () => {
  console.log(`Backend server running at http://localhost:${port}`);
}); 