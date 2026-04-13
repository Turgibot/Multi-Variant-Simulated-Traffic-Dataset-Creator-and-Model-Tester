// API Configuration Service
// Dev: default to backend on :8000. Production build: same-origin (empty) so /api/* goes through edge nginx.
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? 'http://localhost:8000' : '')

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL
  }

  // Generic request method
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    }

    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error('API request failed:', error)
      throw error
    }
  }

  // GET request
  async get(endpoint) {
    return this.request(endpoint, { method: 'GET' })
  }

  // POST request
  async post(endpoint, data) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  // PUT request
  async put(endpoint, data) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data)
    })
  }

  // DELETE request
  async delete(endpoint) {
    return this.request(endpoint, { method: 'DELETE' })
  }

  // Specific API methods
  async getHealth() {
    return this.get('/health')
  }

  async getPublications() {
    return this.get('/api/publications')
  }

  async getTeam() {
    return this.get('/api/team')
  }

  async submitContact(data) {
    return this.post('/api/contact', data)
  }

  // SUMO Simulation API methods
  async getSimulationStatus() {
    return this.get('/api/simulation/status')
  }

  async getDataStatus() {
    return this.get('/api/simulation/data-status')
  }

  async startJourney(start, end) {
    return this.post('/api/simulation/start-journey', { start, end })
  }

  async getVehiclePosition(vehicleId) {
    return this.get(`/api/simulation/vehicle/${vehicleId}/position`)
  }

  async getActiveVehicles() {
    return this.get('/api/simulation/vehicles/active')
  }

  async getFinishedVehicles() {
    return this.get('/api/simulation/vehicles/finished')
  }

  async clearFinishedVehicles() {
    return this.post('/api/simulation/vehicles/finished/clear')
  }

  async startSimulation() {
    return this.post('/api/simulation/start')
  }

  async stopSimulation() {
    return this.post('/api/simulation/stop')
  }

  // Trips playback
  async getCurrentTrips() {
    return this.get('/api/trips/current')
  }

  async nextTripsStep() {
    return this.post('/api/trips/next')
  }

  async startTripsPlayback() {
    return this.post('/api/trips/play')
  }

  async stopTripsPlayback() {
    return this.post('/api/trips/stop')
  }

  async getTripsStatus() {
    return this.get('/api/trips/status')
  }



  async calculateRoute(startX, startY, endX, endY) {
    return this.post(`/api/simulation/calculate-route?start_x=${startX}&start_y=${startY}&end_x=${endX}&end_y=${endY}`)
  }

  async calculateRouteByEdges(startEdge, endEdge) {
    return this.post(`/api/simulation/calculate-route-by-edges?start_edge=${startEdge}&end_edge=${endEdge}`)
  }

  async getSimulationResults(limit = 10) {
    return this.get(`/api/simulation/results?limit=${limit}`)
  }

  async getSimulationStatistics() {
    return this.get('/api/simulation/statistics')
  }

  // Network visualization API methods
  async getNetworkData() {
    return this.get('/api/network/data')
  }

  // Route calculation API methods
  async calculateRouteByEdges(startEdge, endEdge) {
    return this.post('/api/simulation/calculate-route-by-edges', {
      start_edge: startEdge,
      end_edge: endEdge
    })
  }

  async startJourney(startEdge, endEdge, routeEdges) {
    console.log('🚀 API: Starting journey with:', {
      start_edge: startEdge,
      end_edge: endEdge,
      route_edges: routeEdges,
      route_edges_type: typeof routeEdges,
      route_edges_is_array: Array.isArray(routeEdges)
    })
    
    return this.post('/api/simulation/start-journey', {
      start_edge: startEdge,
      end_edge: endEdge,
      route_edges: routeEdges
    })
  }

  async startJourneyDebug(startEdge, endEdge, routeEdges) {
    console.log('🔍 DEBUG API: Testing debug endpoint with:', {
      start_edge: startEdge,
      end_edge: endEdge,
      route_edges: routeEdges,
      route_edges_type: typeof routeEdges,
      route_edges_is_array: Array.isArray(routeEdges)
    })
    
    return this.post('/api/simulation/start-journey-debug', {
      start_edge: startEdge,
      end_edge: endEdge,
      route_edges: routeEdges
    })
  }

  async startJourneyRaw(startEdge, endEdge, routeEdges) {
    console.log('🔍 RAW API: Testing raw endpoint with:', {
      start_edge: startEdge,
      end_edge: endEdge,
      route_edges: routeEdges,
      route_edges_type: typeof routeEdges,
      route_edges_is_array: Array.isArray(routeEdges)
    })
    
    return this.post('/api/simulation/start-journey-raw', {
      start_edge: startEdge,
      end_edge: endEdge,
      route_edges: routeEdges
    })
  }

  async startJourneyValidation(startEdge, endEdge, routeEdges) {
    console.log('🔍 VALIDATION API: Testing validation endpoint with:', {
      start_edge: startEdge,
      end_edge: endEdge,
      route_edges: routeEdges,
      route_edges_type: typeof routeEdges,
      route_edges_is_array: Array.isArray(routeEdges)
    })
    
    return this.post('/api/simulation/start-journey-validation', {
      start_edge: startEdge,
      end_edge: endEdge,
      route_edges: routeEdges
    })
  }

  async startJourneyManual(startEdge, endEdge, routeEdges) {
    console.log('🔍 MANUAL API: Testing manual endpoint with:', {
      start_edge: startEdge,
      end_edge: endEdge,
      route_edges: routeEdges,
      route_edges_type: typeof routeEdges,
      route_edges_is_array: Array.isArray(routeEdges)
    })

    return this.post('/api/simulation/start-journey-manual', {
      start_edge: startEdge,
      end_edge: endEdge,
      route_edges: routeEdges
    })
  }

  // Journey Database API methods

  async saveJourney(journeyData) {
    return this.post('/api/journeys/save', journeyData)
  }

  async getRecentJourneys(limit = 20) {
    return this.get(`/api/journeys/recent?limit=${limit}`)
  }

  async deleteLastJourney() {
    return this.delete('/api/journeys/delete-last')
  }

  async deleteAllJourneys() {
    return this.delete('/api/journeys/delete-all')
  }

  async getJourneyCount() {
    return this.get('/api/journeys/count')
  }

  async getJourneyStatistics() {
    return this.get('/api/journeys/statistics')
  }

  async getDurationVsMaePlotData() {
    return this.get('/api/journeys/plot-data/duration-vs-mae')
  }

  async getDurationVsMaePlotImage() {
    return this.get('/api/journeys/plot-image/duration-vs-mae')
  }

  async getDistanceVsMaePlotData() {
    return this.get('/api/journeys/plot-data/distance-vs-mae')
  }

  async getDistanceVsMaePlotImage() {
    return this.get('/api/journeys/plot-image/distance-vs-mae')
  }

  async getMaeByTimePlotData() {
    return this.get('/api/journeys/plot-data/mae-by-time')
  }

  async getMaeByTimePlotImage() {
    return this.get('/api/journeys/plot-image/mae-by-time')
  }

  async getDurationHistogramPlotData(category) {
    return this.get(`/api/journeys/plot-data/duration-histogram/${category}`)
  }

  async getDurationHistogramPlotImage(category) {
    return this.get(`/api/journeys/plot-image/duration-histogram/${category}`)
  }

  async getDistanceHistogramPlotData(category) {
    return this.get(`/api/journeys/plot-data/distance-histogram/${category}`)
  }

  async getDistanceHistogramPlotImage(category) {
    return this.get(`/api/journeys/plot-image/distance-histogram/${category}`)
  }

  async seedRandomJourneys() {
    return this.post('/api/admin/seed-data')
  }
}

// Export a singleton instance
export default new ApiService()
