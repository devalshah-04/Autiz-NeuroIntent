// Simple WebSocket server that simulates Krishiv's backend
const { WebSocketServer } = require('ws')

// Start server on port 8000
const wss = new WebSocketServer({ port: 8000 })
console.log('Mock WebSocket server running on ws://localhost:8000')

wss.on('connection', (ws) => {
    console.log('Overlay connected!')

    let tick = 0

    // Send fake data every 2 seconds
    const interval = setInterval(() => {
        tick++

        // Every 5th tick, simulate a misread flag
        const misread = tick % 5 === 0

        // Build the exact payload Krishiv's backend will send
        const payload = {
            content_quality_score: parseFloat((Math.random() * 0.4 + 0.6).toFixed(2)),
            prosody_summary: tick % 2 === 0 ? 'flat pitch' : 'steady pace',
            misread_flag: misread,
            misread_description: misread ? 'Monotone delivery may mask confidence' : null,
        }

        // Send as JSON string
        ws.send(JSON.stringify(payload))
        console.log('Sent:', payload)
    }, 2000)

    // Stop sending when overlay disconnects
    ws.on('close', () => {
        clearInterval(interval)
        console.log('Overlay disconnected')
    })
})