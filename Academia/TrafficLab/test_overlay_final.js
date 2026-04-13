#!/usr/bin/env node
import fetch from 'node-fetch';

const API_BASE = 'http://localhost:8000/api';

async function testOverlay() {
  try {
    console.log('🧪 Testing Finished Vehicle Overlay...\n');
    
    // Start simulation
    console.log('1. Starting simulation...');
    const startResponse = await fetch(`${API_BASE}/simulation/start`, { method: 'POST' });
    const startData = await startResponse.json();
    console.log('   Response:', startData);
    
    // Wait a moment for simulation to start
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Add a vehicle
    console.log('\n2. Adding a vehicle...');
    const vehicleData = {
      start_point: 'AT0AS0',
      destination_point: 'AT1AU1'
    };
    
    const addResponse = await fetch(`${API_BASE}/simulation/vehicles/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(vehicleData)
    });
    const addData = await addResponse.json();
    console.log('   Response:', addData);
    
    // Poll for finished vehicles
    console.log('\n3. Polling for finished vehicles...');
    let attempts = 0;
    const maxAttempts = 30; // 30 seconds
    
    while (attempts < maxAttempts) {
      const finishedResponse = await fetch(`${API_BASE}/simulation/vehicles/finished`);
      const finishedData = await finishedResponse.json();
      
      console.log(`   Attempt ${attempts + 1}: Found ${finishedData.finished_vehicles?.length || 0} finished vehicles`);
      
      if (finishedData.finished_vehicles && finishedData.finished_vehicles.length > 0) {
        console.log('   ✅ Finished vehicles detected!');
        console.log('   Vehicle details:', finishedData.finished_vehicles[0]);
        console.log('\n🎯 The overlay should now be visible in the frontend!');
        console.log('   - Check http://localhost:3000');
        console.log('   - The overlay should appear in the center of the map');
        console.log('   - It should auto-close after 20 seconds or can be closed manually');
        break;
      }
      
      attempts++;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    if (attempts >= maxAttempts) {
      console.log('   ⚠️  No finished vehicles detected within 30 seconds');
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

testOverlay();
