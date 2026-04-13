<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Demo Header -->
    <div class="bg-gradient-to-r from-blue-600 to-blue-800 text-white py-5">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center">
          <router-link to="/" class="text-2xl font-bold hover:text-blue-200">
            SmartTransportation Lab
          </router-link>
          <h1 class="text-xl">ETA Prediction Performance Testing</h1>
        </div>
      </div>
    </div>

    <!-- Main Demo Content -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div class="flex flex-col lg:flex-row gap-6">
              
        <!-- Network Map (Full width) -->
        <div class="w-full bg-white rounded-lg shadow-lg">
          <!-- Simulation Map -->
          <div class="relative h-full">
            <div class="bg-gray-100 rounded-t-lg p-4 border-b">
              <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div class="flex-1">
                  <h2 class="text-xl font-semibold">Traffic Network Map</h2>
                  <div class="text-sm text-gray-600 mt-1">
                    Click on roads to set start and destination points
                  </div>
                </div>
                
                <!-- Center Time Display -->
                <div class="absolute left-1/2 transform -translate-x-1/2 sm:relative sm:left-auto sm:transform-none">
                  <div class="text-xl font-semibold text-gray-800 text-center sm:text-left">
                    {{ formatTimeWithDay(simulationTime) }}
                  </div>
                </div>
                
                <!-- Control Buttons -->
                <div class="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 sm:space-x-3 sm:gap-0">
                  <!-- Simulation Playback Controls - Always show -->
                  <div class="flex items-center space-x-2">
                    <!-- Reset Points Button - Show when points are selected but journey not running -->
                        <button 
                      v-if="startPoint && !isJourneyRunning"
                      @click="resetPoints"
                      class="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-semibold transition duration-300 text-sm"
                        >
                      🔄 Reset Points
                        </button>
                    
                        <button 
                      @click="handleMainButtonClick()"
                      :class="getMainButtonClass()"
                      class="text-white px-4 py-2 rounded-lg font-semibold transition duration-300 text-sm"
                        >
                      {{ getMainButtonText() }}
                        </button>
                    
                    <div class="text-sm text-gray-600">
                      Trips: {{ simulationStatus.trips_added || 0 }}
                      </div>
                    <div class="text-sm text-gray-600">
                      Active: {{ simulationStatus.vehicles_in_route || 0 }}
                    </div>
                  </div>
                    </div>
                  </div>
                </div>
            
            <!-- SUMO Network Container -->
                  <div 
                    class="relative bg-gray-300 h-[calc(100vh-200px)] sm:h-[calc(105.8vh-317.4px)]"
                    ref="mapContainer"
                  >
              <!-- SUMO Network Visualization -->
              <svg 
                ref="networkSvg"
                :viewBox="svgViewBox" 
                width="100%" 
                height="100%" 
                class="absolute inset-0 cursor-crosshair map-container touch-none" 
                style="background: #f1f5f9; min-height: 690px;"
                preserveAspectRatio="xMidYMid meet"
                @mousemove="handleMouseMove"
                @click="handleMapClick"
                @touchstart="handleTouchStart"
                @touchmove="handleTouchMove"
                @touchend="handleTouchEnd"
              >
                <!-- SUMO Network Edges (Roads) -->
                <g v-if="networkData && networkData.edges">
                  <template v-for="edge in networkData.edges" :key="edge.id">
                    <!-- Skip rendering E5 and E6 edges only -->
                    <template v-if="edge.id === 'E5' || edge.id === 'E6'"></template>
                    <template v-else>
                    <!-- Use path for express edges with detailed shape, line for regular roads -->
                        <path 
                      v-if="isExpressEdge(edge) && edge.shape_points && edge.shape_points.length > 0 && getPathData(edge.shape_points)"
                      :d="getPathData(edge.shape_points)"
                          :stroke="getRouteStrokeColor(edge)"
                          :stroke-width="getRouteStrokeWidth(edge)"
                          :opacity="getRoadOpacity(edge)"
                          :class="getRoadClass(edge)"
                      :data-road-name="edge.id"
                          fill="none"
                      @click="handleEdgeClick(edge, $event)"
                    />
                    <!-- Fallback to line if path data is invalid -->
                    <line 
                      v-else-if="isExpressEdge(edge)"
                      :x1="getJunctionPosition(edge.from_junction)?.x || 0"
                      :y1="getJunctionPosition(edge.from_junction)?.y || 0"
                      :x2="getJunctionPosition(edge.to_junction)?.x || 0"
                      :y2="getJunctionPosition(edge.to_junction)?.y || 0"
                      :stroke="getRouteStrokeColor(edge)"
                      :stroke-width="getRouteStrokeWidth(edge)"
                          :opacity="getRoadOpacity(edge)"
                          :class="getRoadClass(edge)"
                      :data-road-name="edge.id"
                      @click="handleEdgeClick(edge, $event)"
                    />
                    <!-- Simple line representation for regular roads -->
                    <line 
                      v-else
                      :x1="getJunctionPosition(edge.from_junction)?.x || 0"
                      :y1="getJunctionPosition(edge.from_junction)?.y || 0"
                      :x2="getJunctionPosition(edge.to_junction)?.x || 0"
                      :y2="getJunctionPosition(edge.to_junction)?.y || 0"
                      :stroke="getRouteStrokeColor(edge)"
                      :stroke-width="getRouteStrokeWidth(edge)"
                      :opacity="getRoadOpacity(edge)"
                      :class="getRoadClass(edge)"
                      :data-road-name="edge.id"
                      @click="handleEdgeClick(edge, $event)"
                    />
                    
                    <!-- Invisible clickable areas for easier selection -->
                        <path 
                      v-if="isExpressEdge(edge) && edge.shape_points && edge.shape_points.length > 0 && getPathData(edge.shape_points)"
                      :d="getPathData(edge.shape_points)"
                          stroke="transparent" 
                      :stroke-width="60"
                          fill="none"
                          :class="getRoadClass(edge)"
                      :data-road-name="edge.id"
                      @click="handleEdgeClick(edge, $event)"
                    />
                    <!-- Fallback invisible clickable line -->
                    <line 
                      v-else-if="isExpressEdge(edge)"
                      :x1="getJunctionPosition(edge.from_junction)?.x || 0"
                      :y1="getJunctionPosition(edge.from_junction)?.y || 0"
                      :x2="getJunctionPosition(edge.to_junction)?.x || 0"
                      :y2="getJunctionPosition(edge.to_junction)?.y || 0"
                      stroke="transparent"
                      :stroke-width="60"
                      :class="getRoadClass(edge)"
                      :data-road-name="edge.id"
                      @click="handleEdgeClick(edge, $event)"
                    />
                    <line 
                      v-else
                      :x1="getJunctionPosition(edge.from_junction)?.x || 0"
                      :y1="getJunctionPosition(edge.from_junction)?.y || 0"
                      :x2="getJunctionPosition(edge.to_junction)?.x || 0"
                      :y2="getJunctionPosition(edge.to_junction)?.y || 0"
                      stroke="transparent"
                      :stroke-width="60"
                      :class="getRoadClass(edge)"
                      :data-road-name="edge.id"
                      @click="handleEdgeClick(edge, $event)"
                        />
                    </template>
                  </template>
                </g>
                
                <!-- SUMO Network Junctions -->
                <g v-if="networkData && networkData.junctions">
                  <circle 
                    v-for="junction in networkData.junctions" 
                    :key="junction.id"
                    :cx="junction.x" 
                    :cy="junction.y" 
                    r="1.5" 
                    fill="#1f2937"
                    opacity="0.6"
                  />
                </g>
                
                
                <!-- Start point marker -->
                <circle 
                  v-if="startPoint" 
                  :cx="startPoint.x" 
                  :cy="startPoint.y" 
                  r="100" 
                  fill="#10b981" 
                  stroke="white" 
                  stroke-width="8"
                  class="cursor-pointer drop-shadow-lg"
                />
                <text 
                  v-if="startPoint" 
                  :x="startPoint.x" 
                  :y="startPoint.y + 60" 
                  text-anchor="middle" 
                  fill="black" 
                  font-size="180" 
                  font-weight="bold"
                >
                  S
                </text>
                
                
                <!-- Destination point marker -->
                <circle 
                  v-if="destinationPoint" 
                  :cx="destinationPoint.x" 
                  :cy="destinationPoint.y" 
                  r="100" 
                  fill="#ef4444" 
                  stroke="white" 
                  stroke-width="8"
                  class="cursor-pointer drop-shadow-lg"
                />
                <text 
                  v-if="destinationPoint" 
                  :x="destinationPoint.x" 
                  :y="destinationPoint.y + 60" 
                  text-anchor="middle" 
                  fill="black" 
                  font-size="180" 
                  font-weight="bold"
                >
                  D
                </text>
                
                
                <!-- Route path -->
                <path 
                  v-if="routePath" 
                  :d="routePath" 
                  stroke="#00ff00" 
                  stroke-width="8" 
                  fill="none" 
                  opacity="0.9"
                  class="route-path"
                />
                
                <!-- Real Vehicles from Simulation -->
                <g v-if="activeVehicles && activeVehicles.length > 0">
                <!-- Regular vehicles (circles) -->
                <circle 
                    v-for="vehicle in activeVehicles.filter(v => !isNaN(v.x) && !isNaN(v.y) && v.type !== 'user_defined')"
                    :key="vehicle.id"
                    :cx="vehicle.x"
                    :cy="vehicle.y"
                    r="36"
                    :fill="getVehicleColor(vehicle.type, vehicle.status)"
                    stroke="#000000"
                    stroke-width="3"
                    class="vehicle-marker"
                    opacity="0.8"
                />
                <!-- User-defined vehicles (yellow stars) -->
                <path 
                    v-for="vehicle in activeVehicles.filter(v => !isNaN(v.x) && !isNaN(v.y) && v.type === 'user_defined')"
                    :key="vehicle.id"
                    :d="getStarPath(vehicle.x, vehicle.y, 108)"
                    :fill="getVehicleColor(vehicle.type, vehicle.status)"
                    stroke="#000000"
                    stroke-width="9"
                    class="vehicle-marker"
                    opacity="0.8"
                />
                </g>
                
              </svg>

              <!-- Map Legend -->
              <div v-if="showLegend" class="absolute top-2 left-2 sm:top-4 sm:left-4 bg-white bg-opacity-95 rounded-lg shadow-lg pointer-events-auto animate-fade-in" style="max-width: 180px;">
                <!-- Legend Header (Always Visible) -->
                <div 
                  @click="toggleLegend"
                  class="flex items-center justify-between cursor-pointer hover:bg-gray-50 rounded-t-lg transition-colors"
                  :class="legendCollapsed ? 'p-2' : 'p-3'"
                >
                  <h3 class="font-bold text-gray-800 flex items-center"
                      :class="legendCollapsed ? 'text-xs' : 'text-sm'">
                    <span :class="legendCollapsed ? 'mr-0.5' : 'mr-1'">🗺️</span>
                    <span v-if="!legendCollapsed">Legend</span>
                  </h3>
                  <span class="text-gray-500 text-xs transform transition-transform" 
                        :class="{ 'rotate-180': !legendCollapsed }">
                    ▼
                  </span>
                </div>
                
                <!-- Collapsible Content -->
                <div v-if="!legendCollapsed" class="px-3 pb-3">
                  <!-- Roads Section -->
                  <div class="mb-3">
                    <h4 class="text-xs font-semibold text-gray-700 mb-1">Roads</h4>
                    <div class="space-y-1">
                      <div class="flex items-center">
                        <div class="w-4 h-0.5 bg-gray-600 mr-2"></div>
                        <span class="text-xs text-gray-600">Regular</span>
                      </div>
                      <div class="flex items-center">
                        <svg class="w-4 h-0.5 mr-2" viewBox="0 0 16 2">
                          <line x1="0" y1="1" x2="16" y2="1" stroke="#1e40af" stroke-width="1" opacity="0.6" stroke-dasharray="2,2"/>
                        </svg>
                        <span class="text-xs text-gray-600">Expressways</span>
                      </div>
                      <div class="flex items-center">
                        <div class="w-4 h-1 bg-green-500 mr-2"></div>
                        <span class="text-xs text-gray-600">Route</span>
                      </div>
                    </div>
                  </div>

                  <!-- Markers Section -->
                  <div class="mb-3">
                    <h4 class="text-xs font-semibold text-gray-700 mb-1">Markers</h4>
                    <div class="space-y-1">
                      <div class="flex items-center">
                        <div class="w-3 h-3 bg-green-500 rounded-full mr-2 border border-white"></div>
                        <span class="text-xs text-gray-600">Start (S)</span>
                      </div>
                      <div class="flex items-center">
                        <div class="w-3 h-3 bg-red-500 rounded-full mr-2 border border-white"></div>
                        <span class="text-xs text-gray-600">Dest (D)</span>
                      </div>
                    </div>
                  </div>

                  <!-- Vehicles Section -->
                  <div class="mb-3">
                    <h4 class="text-xs font-semibold text-gray-700 mb-1">Vehicles</h4>
                    <div class="space-y-1">
                      <div class="flex items-center">
                        <div class="w-3 h-3 bg-blue-500 rounded-full mr-2 border border-black"></div>
                        <span class="text-xs text-gray-600">Cars</span>
                      </div>
                      <div class="flex items-center">
                        <div class="w-3 h-3 bg-orange-500 rounded-full mr-2 border border-black"></div>
                        <span class="text-xs text-gray-600">Buses</span>
                      </div>
                      <div class="flex items-center">
                        <div class="w-3 h-3 bg-green-500 rounded-full mr-2 border border-black"></div>
                        <span class="text-xs text-gray-600">Trucks</span>
                      </div>
                      <div class="flex items-center">
                        <svg class="w-3 h-3 mr-2" viewBox="0 0 12 12">
                          <path d="M6 1l1.5 3L12 4.5l-2.5 2.5L11 11l-3-1.5L5 11l1-4L3 4.5l3.5-.5L6 1z" fill="#fbbf24" stroke="#000000" stroke-width="0.3"/>
                        </svg>
                        <span class="text-xs text-gray-600">Your Vehicle</span>
                      </div>
                      <div class="flex items-center">
                        <div class="w-3 h-3 bg-white rounded-full mr-2 border border-black"></div>
                        <span class="text-xs text-gray-600">Not Tracked</span>
                      </div>
                    </div>
                  </div>

                </div>
              </div>

              <!-- Statistics Section (Left side overlay) -->
              <div class="absolute top-0 right-full bg-slate-900 bg-opacity-98 rounded-l-lg shadow-xl pointer-events-auto animate-fade-in h-[calc(100vh-200px)] sm:h-[calc(105.8vh-317.4px)] w-[250px] sm:w-[280px]" style="overflow-y: auto; scrollbar-width: none; -ms-overflow-style: none;">
                <!-- Statistics Header -->
                <div class="flex items-center justify-between p-4 border-b border-slate-700 bg-slate-800">
                  <h3 class="text-sm font-semibold text-slate-100 flex items-center">
                    <span class="mr-2 text-green-400">📈</span>
                    Model Performance Analysis
                  </h3>
                  <button 
                    @click="generatePDF" 
                    class="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg text-xs font-medium transition-colors duration-200 flex items-center"
                    :disabled="isGeneratingPDF"
                  >
                    <svg v-if="!isGeneratingPDF" class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                    </svg>
                    <svg v-else class="w-3 h-3 mr-1 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                    </svg>
                    {{ isGeneratingPDF ? 'Generating...' : 'Export PDF' }}
                  </button>
                </div>
                
                <!-- Statistics Content -->
                <div class="p-4 h-full overflow-y-auto hide-scrollbar scroll-smooth" style="scrollbar-width: none; -ms-overflow-style: none;">
                  <!-- First Line: Basic Statistics -->
                  <div class="grid grid-cols-3 gap-2 mb-3">
                    <!-- Total Journeys -->
                    <div class="bg-slate-700/50 rounded-lg p-2 text-center">
                      <div class="text-xs text-slate-400 mb-1 font-medium">Total Journeys</div>
                      <div class="text-sm font-bold text-slate-100">{{ journeyStatistics.total_journeys || 0 }}</div>
                    </div>
                    
                    <!-- Average Duration -->
                    <div class="bg-slate-700/50 rounded-lg p-2 text-center">
                      <div class="text-xs text-slate-400 mb-1 font-medium">Avg Duration</div>
                      <div class="text-sm font-bold text-slate-100">{{ formatTime(journeyStatistics.average_duration || 0) }}</div>
                    </div>
                    
                    <!-- Average Distance -->
                    <div class="bg-slate-700/50 rounded-lg p-2 text-center">
                      <div class="text-xs text-slate-400 mb-1 font-medium">Avg Distance</div>
                      <div class="text-sm font-bold text-slate-100">{{ ((journeyStatistics.average_distance || 0) / 1000).toFixed(1) }}km</div>
                    </div>
                  </div>
                  
                  <!-- Second Line: Prediction Accuracy Metrics -->
                  <div class="grid grid-cols-3 gap-2 mb-3">
                    <!-- MAE -->
                    <div class="bg-slate-700/50 rounded-lg p-2 text-center">
                      <div class="text-xs text-slate-400 mb-1 font-medium">MAE</div>
                      <div class="text-sm font-bold text-slate-100">{{ journeyStatistics.mae || 0 }}s</div>
                    </div>
                    
                    <!-- RMSE -->
                    <div class="bg-slate-700/50 rounded-lg p-2 text-center">
                      <div class="text-xs text-slate-400 mb-1 font-medium">RMSE</div>
                      <div class="text-sm font-bold text-slate-100">{{ journeyStatistics.rmse || 0 }}s</div>
                    </div>
                    
                    <!-- MAPE -->
                    <div class="bg-slate-700/50 rounded-lg p-2 text-center">
                      <div class="text-xs text-slate-400 mb-1 font-medium">MAPE</div>
                      <div class="text-sm font-bold text-slate-100">{{ (journeyStatistics.mape || 0).toFixed(1) }}%</div>
                    </div>
                  </div>
                  
                  <!-- Separating Line -->
                  <div class="border-t border-slate-600 mb-4"></div>
                  
                  <!-- Third Line: MAE by trip duration -->
                  <div class="mb-3">
                    <div class="text-xs text-slate-400 mb-2 font-medium text-center">MAE by trip duration</div>
                    <div class="grid grid-cols-3 gap-2">
                      <!-- Short Trips -->
                      <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-help" title="Duration: less than 278 seconds">
                        <div class="text-xs text-slate-400 mb-1 font-medium">Short</div>
                        <div class="text-sm font-bold text-slate-100">{{ journeyStatistics.short_trips?.mae || 0 }}s</div>
                        <div class="text-xs text-slate-400">{{ journeyStatistics.short_trips?.count || 0 }}</div>
                      </div>
                      
                      <!-- Medium Trips -->
                      <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-help" title="Duration: 278 to 609 seconds">
                        <div class="text-xs text-slate-400 mb-1 font-medium">Medium</div>
                        <div class="text-sm font-bold text-slate-100">{{ journeyStatistics.medium_trips?.mae || 0 }}s</div>
                        <div class="text-xs text-slate-400">{{ journeyStatistics.medium_trips?.count || 0 }}</div>
                      </div>
                      
                      <!-- Long Trips -->
                      <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-help" title="Duration: more than 609 seconds">
                        <div class="text-xs text-slate-400 mb-1 font-medium">Long</div>
                        <div class="text-sm font-bold text-slate-100">{{ journeyStatistics.long_trips?.mae || 0 }}s</div>
                        <div class="text-xs text-slate-400">{{ journeyStatistics.long_trips?.count || 0 }}</div>
                      </div>
                    </div>
                  </div>
                  
                  <!-- Separating Line -->
                  <div class="border-t border-slate-600 mb-4"></div>
                  
                  <!-- Fourth Line: MAE by trip distance -->
                  <div class="mb-3">
                    <div class="text-xs text-slate-400 mb-2 font-medium text-center">MAE by trip distance</div>
                    <div class="grid grid-cols-3 gap-2">
                      <!-- Short Distance -->
                      <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-help" title="Distance: less than 4 kilometers">
                        <div class="text-xs text-slate-400 mb-1 font-medium">Short</div>
                        <div class="text-sm font-bold text-slate-100">{{ journeyStatistics.short_trips_distance?.mae || 0 }}s</div>
                        <div class="text-xs text-slate-400">{{ journeyStatistics.short_trips_distance?.count || 0 }}</div>
                      </div>
                      
                      <!-- Medium Distance -->
                      <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-help" title="Distance: 4 to 11 kilometers">
                        <div class="text-xs text-slate-400 mb-1 font-medium">Medium</div>
                        <div class="text-sm font-bold text-slate-100">{{ journeyStatistics.medium_trips_distance?.mae || 0 }}s</div>
                        <div class="text-xs text-slate-400">{{ journeyStatistics.medium_trips_distance?.count || 0 }}</div>
                      </div>
                      
                      <!-- Long Distance -->
                      <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-help" title="Distance: more than 11 kilometers">
                        <div class="text-xs text-slate-400 mb-1 font-medium">Long</div>
                        <div class="text-sm font-bold text-slate-100">{{ journeyStatistics.long_trips_distance?.mae || 0 }}s</div>
                        <div class="text-xs text-slate-400">{{ journeyStatistics.long_trips_distance?.count || 0 }}</div>
                      </div>
                    </div>
                  </div>
                  
                  <!-- Separating Line -->
                  <div class="border-t border-slate-600 mb-4"></div>
                  
                  <!-- Fourth Section: Visual Analysis -->
                  <div class="mb-3">
                    <div class="text-xs text-slate-400 mb-2 font-medium text-center">Visual Analysis</div>
                    
                    <!-- First Line: Scatter Plots and Time Analysis -->
                    <div class="grid grid-cols-3 gap-2 mb-3">
                      <!-- Trip Duration vs MAE Scatter Plot -->
                      <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-pointer hover:bg-slate-600/50 transition-colors" 
                           @click="openPlotWindow('duration-vs-mae-scatter')">
                        <div class="text-xs text-slate-400 mb-1 font-medium">Trip Duration vs MAE</div>
                        <div class="text-xs text-slate-300">Scatter Plot</div>
                        <div class="text-xs text-slate-500 mt-1">📊</div>
                      </div>
                      
                      <!-- Trip Distance vs MAE Scatter Plot -->
                      <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-pointer hover:bg-slate-600/50 transition-colors" 
                           @click="openPlotWindow('distance-vs-mae-scatter')">
                        <div class="text-xs text-slate-400 mb-1 font-medium">Trip Distance vs MAE</div>
                        <div class="text-xs text-slate-300">Scatter Plot</div>
                        <div class="text-xs text-slate-500 mt-1">📊</div>
                      </div>
                      
                      <!-- MAE by Time of Day -->
                      <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-pointer hover:bg-slate-600/50 transition-colors" 
                           @click="openPlotWindow('mae-by-time')">
                        <div class="text-xs text-slate-400 mb-1 font-medium">MAE by Time of Day</div>
                        <div class="text-xs text-slate-300">Bar Chart</div>
                        <div class="text-xs text-slate-500 mt-1">📊</div>
                      </div>
                    </div>
                    
                    <!-- Second Line: Trip Duration vs MAE Histograms -->
                    <div class="mb-2">
                      <div class="text-xs text-slate-400 mb-2 font-medium text-center">Trip Duration vs MAE Histogram</div>
                      <div class="grid grid-cols-3 gap-2">
                        <!-- Short Trips Duration Histogram -->
                        <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-pointer hover:bg-slate-600/50 transition-colors" 
                             @click="openPlotWindow('duration-histogram-short')">
                          <div class="text-xs text-slate-400 mb-1 font-medium">Short</div>
                          <div class="text-xs text-slate-300">Duration</div>
                          <div class="text-xs text-slate-500 mt-1">📊</div>
                        </div>
                        
                        <!-- Medium Trips Duration Histogram -->
                        <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-pointer hover:bg-slate-600/50 transition-colors" 
                             @click="openPlotWindow('duration-histogram-medium')">
                          <div class="text-xs text-slate-400 mb-1 font-medium">Medium</div>
                          <div class="text-xs text-slate-300">Duration</div>
                          <div class="text-xs text-slate-500 mt-1">📊</div>
                        </div>
                        
                        <!-- Long Trips Duration Histogram -->
                        <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-pointer hover:bg-slate-600/50 transition-colors" 
                             @click="openPlotWindow('duration-histogram-long')">
                          <div class="text-xs text-slate-400 mb-1 font-medium">Long</div>
                          <div class="text-xs text-slate-300">Duration</div>
                          <div class="text-xs text-slate-500 mt-1">📊</div>
                        </div>
                      </div>
                    </div>
                    
                    <!-- Third Line: Trip Distance vs MAE Histograms -->
                    <div class="mb-3">
                      <div class="text-xs text-slate-400 mb-2 font-medium text-center">Trip Distance vs MAE Histogram</div>
                      <div class="grid grid-cols-3 gap-2">
                        <!-- Short Trips Distance Histogram -->
                        <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-pointer hover:bg-slate-600/50 transition-colors" 
                             @click="openPlotWindow('distance-histogram-short')">
                          <div class="text-xs text-slate-400 mb-1 font-medium">Short</div>
                          <div class="text-xs text-slate-300">Distance</div>
                          <div class="text-xs text-slate-500 mt-1">📊</div>
                        </div>
                        
                        <!-- Medium Trips Distance Histogram -->
                        <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-pointer hover:bg-slate-600/50 transition-colors" 
                             @click="openPlotWindow('distance-histogram-medium')">
                          <div class="text-xs text-slate-400 mb-1 font-medium">Medium</div>
                          <div class="text-xs text-slate-300">Distance</div>
                          <div class="text-xs text-slate-500 mt-1">📊</div>
                        </div>
                        
                        <!-- Long Trips Distance Histogram -->
                        <div class="bg-slate-700/50 rounded-lg p-2 text-center cursor-pointer hover:bg-slate-600/50 transition-colors" 
                             @click="openPlotWindow('distance-histogram-long')">
                          <div class="text-xs text-slate-400 mb-1 font-medium">Long</div>
                          <div class="text-xs text-slate-300">Distance</div>
                          <div class="text-xs text-slate-500 mt-1">📊</div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <!-- Additional Statistics Placeholder -->
                  <div class="text-center text-slate-400 text-xs py-4">
                    <div class="text-2xl mb-2">📈</div>
                    <div class="font-medium">More analytics coming soon</div>
                    <div class="text-xs mt-1">Advanced performance metrics will be added here</div>
                  </div>
                </div>
              </div>

              <!-- Results Summary Section -->
              <div class="absolute top-0 left-full bg-slate-900 bg-opacity-98 rounded-r-lg shadow-xl pointer-events-auto animate-fade-in h-[calc(105.8vh-317.4px)]" style="width: 280px; overflow-y: auto; scrollbar-width: none; -ms-overflow-style: none;">
                <!-- Results Header -->
                <div class="flex items-center justify-between p-4 border-b border-slate-700 bg-slate-800">
                  <h3 class="text-sm font-semibold text-slate-100 flex items-center">
                    <span class="mr-2 text-blue-400">📊</span>
                    Recent Journeys
                  </h3>
                </div>
                
                <!-- Results Content -->
                <div class="p-4 h-full overflow-y-auto hide-scrollbar scroll-smooth" style="scrollbar-width: none; -ms-overflow-style: none;">
                  <div v-if="vehicleResults.length === 0" class="text-center text-slate-400 text-sm py-8">
                    <div class="text-4xl mb-3">🚗</div>
                    <div class="font-medium">No journeys started yet</div>
                    <div class="text-xs mt-1">Click on roads to set start and destination points</div>
                  </div>
                  
                  <div v-else class="space-y-1.5">
                    <div v-for="(result, index) in vehicleResults" :key="result.vehicle_id" 
                         class="border border-slate-600 rounded-lg p-2 bg-slate-800 hover:bg-slate-750 transition-all duration-200 hover:shadow-lg hover:border-slate-500 will-change-transform"
                         :class="{ 'opacity-50': result.status === 'running' && !isSimulationPlaying }">
                      <!-- Status Header -->
                      <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center space-x-2">
                          <span class="text-xs font-medium text-slate-300">Journey #{{ totalJourneyCount - index }}</span>
                        </div>
                        <span class="text-xs px-3 py-1 rounded-full font-medium flex items-center space-x-1"
                              :class="result.status === 'finished' ? 'bg-emerald-600 text-emerald-100' : 'bg-blue-600 text-blue-100'">
                          <span v-if="result.status === 'finished'">✓</span>
                          <span class="font-mono">{{ result.status === 'finished' ? 'Completed' : 'Running' }}</span>
                          <span v-if="result.status === 'running'" class="ml-1 font-mono text-xs">
                            {{ formatTime(Math.max(0, simulationTime - result.start_time)) }}
                          </span>
                        </span>
                      </div>
                      
                      <!-- Top Section: Start and Distance -->
                      <div class="grid grid-cols-2 gap-1.5 mb-2">
                        <!-- Start Time -->
                        <div class="bg-slate-700/50 rounded p-1.5">
                          <div class="text-xs text-slate-400 mb-0.5 font-medium flex items-center">
                            <span class="mr-1">🕐</span>
                            Start
                          </div>
                          <div class="text-xs font-mono text-slate-100">
                            {{ result.start_time_string ? formatTimeWithDay(result.start_time_string) : formatTimeWithDay(result.start_time) }}
                          </div>
                        </div>
                        
                        <!-- Distance -->
                        <div class="bg-slate-700/50 rounded p-1.5">
                          <div class="text-xs text-slate-400 mb-0.5 font-medium flex items-center">
                            <span class="mr-1">📏</span>
                            Distance
                          </div>
                          <div class="text-xs font-mono text-slate-100">
                            {{ (result.distance / 1000).toFixed(2) }}km
                          </div>
                        </div>
                      </div>
                      
                      <!-- Line Separator -->
                      <div class="border-t border-slate-600 mb-2"></div>
                      
                      <!-- Middle Section: Prediction -->
                      <div class="text-xs font-medium text-slate-300 mb-1.5 flex items-center">
                        <span class="mr-1">🎯</span>
                        Prediction
                      </div>
                      <div class="grid grid-cols-2 gap-1.5 mb-2">
                        <!-- ETA -->
                        <div class="bg-slate-700/50 rounded p-1.5">
                          <div class="text-xs text-slate-400 mb-0.5 font-medium flex items-center">
                            <span class="mr-1">🎯</span>
                            ETA
                          </div>
                          <div class="text-xs font-mono text-slate-100">
                            {{ formatTimeModulo24(result.predicted_eta) }}
                          </div>
                        </div>
                        
                        <!-- Estimated Duration -->
                        <div class="bg-slate-700/50 rounded p-1.5">
                          <div class="text-xs text-slate-400 mb-0.5 font-medium flex items-center">
                            <span class="mr-1">⏱️</span>
                            Est. Duration
                          </div>
                          <div class="text-xs font-mono text-slate-100">
                            {{ calculateDuration(result.start_time_string, formatTime(result.predicted_eta)) }}
                          </div>
                        </div>
                      </div>
                      
                      <!-- Line Separator -->
                      <div class="border-t border-slate-600 mb-2"></div>
                      
                      <!-- Results (only if finished) -->
                      <div v-if="result.status === 'finished'" class="space-y-1.5">
                        <div class="text-xs font-medium text-slate-300 mb-1.5 flex items-center">
                          <span class="mr-1">📊</span>
                          Results
                        </div>
                        
                        <!-- First Line: Arrival Time and Duration -->
                        <div class="grid grid-cols-2 gap-1.5 mb-1.5">
                          <!-- Arrival Time -->
                          <div class="bg-slate-700/50 rounded p-1.5">
                            <div class="text-xs text-slate-400 mb-0.5 font-medium">Arrival Time</div>
                            <div class="text-xs font-mono text-slate-100">
                              {{ result.end_time_string ? formatTimeModulo24(result.end_time_string) : formatTimeModulo24(result.end_time) }}
                            </div>
                          </div>
                          
                          <!-- Duration -->
                          <div class="bg-slate-700/50 rounded p-1.5">
                            <div class="text-xs text-slate-400 mb-0.5 font-medium">Duration</div>
                            <div class="text-xs font-mono text-slate-100">
                              {{ formatTime(result.actual_duration) }}
                            </div>
                          </div>
                        </div>
                        
                        <!-- Second Line: Error and Accuracy -->
                        <div class="grid grid-cols-2 gap-1.5">
                          <!-- Absolute Error -->
                          <div class="bg-slate-700/50 rounded p-1.5">
                            <div class="text-xs text-slate-400 mb-0.5 font-medium">Error</div>
                            <div class="text-xs font-mono px-1.5 py-0.5 rounded" 
                                 :class="calculateError(result) < 30 ? 'text-emerald-300 bg-emerald-900/30' : calculateError(result) < 60 ? 'text-yellow-300 bg-yellow-900/30' : 'text-red-300 bg-red-900/30'">
                              {{ Math.round(calculateError(result)) }}s
                            </div>
                          </div>
                          
                          <!-- Accuracy -->
                          <div class="bg-slate-700/50 rounded p-1.5">
                            <div class="text-xs text-slate-400 mb-0.5 font-medium">Accuracy</div>
                            <div class="text-xs font-mono px-1.5 py-0.5 rounded" 
                                 :class="calculateAccuracy(result) > 80 ? 'text-emerald-300 bg-emerald-900/30' : calculateAccuracy(result) > 60 ? 'text-yellow-300 bg-yellow-900/30' : 'text-red-300 bg-red-900/30'">
                              {{ calculateAccuracy(result).toFixed(1) }}%
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <!-- Scroll to top button -->
                    <div v-if="vehicleResults.length > 3" class="sticky bottom-0 pt-2">
                      <button 
                        @click="scrollToTop"
                        class="w-full bg-slate-700 hover:bg-slate-600 text-slate-300 hover:text-slate-100 py-1.5 px-3 rounded-lg text-xs font-medium transition-colors duration-200"
                      >
                        ↑ Scroll to Top
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Expressway Overlay - Elegant design -->
              <div v-if="showExpresswayOverlay" class="fixed top-4 right-4 z-50 pointer-events-auto">
                <div class="bg-red-500 bg-opacity-95 rounded-xl shadow-xl p-4 text-center max-w-xs animate-pulse">
                  <div class="text-2xl mb-2 animate-bounce">
                    <span>🚫</span>
                </div>
                  <h3 class="text-lg font-bold text-white mb-1">
                    Expressway Not Allowed
                  </h3>
                  <p class="text-sm text-red-100">
                    Start and destination points cannot be set on expressways
                  </p>
              </div>
              </div>
              

              <!-- Instruction Overlay - Only show when simulation is stopped -->
              <div v-if="!isSimulationPlaying" class="absolute top-4 left-4 pointer-events-none">
                <div class="bg-white bg-opacity-95 rounded-xl shadow-xl p-4 text-center max-w-xs animate-pulse">
                  <div class="text-2xl mb-2 animate-bounce">
                    <span v-if="!startPoint">📍</span>
                    <span v-else-if="!destinationPoint">🎯</span>
                    <span v-else>🚀</span>
                  </div>
                  <h3 class="text-lg font-bold text-gray-800 mb-1">
                    <span v-if="!startPoint">Set Starting Point</span>
                    <span v-else-if="!destinationPoint">Set Destination</span>
                    <span v-else>Ready to Start!</span>
                  </h3>
                  <p class="text-sm text-gray-600">
                    {{ getInstructionText() }}
                  </p>
                </div>
              </div>
              
              <!-- Invalid Click Message -->
              <div v-if="invalidClickMessage" class="absolute top-4 right-4 pointer-events-none">
                <div class="bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg animate-pulse">
                  <div class="flex items-center">
                    <span class="text-lg mr-2">⚠️</span>
                    <span class="text-sm font-medium">{{ invalidClickMessage }}</span>
                  </div>
                </div>
              </div>
              
              <!-- Simulation Running Overlay - Same dimensions as instruction overlay -->
            <div v-if="showSimulationOverlay" class="absolute top-4 right-4 pointer-events-auto">
              <div class="bg-orange-500 bg-opacity-95 rounded-xl shadow-xl p-4 text-center max-w-xs animate-pulse">
                <div class="text-2xl mb-2 animate-bounce">
                  <span>🚗</span>
              </div>
                <h3 class="text-lg font-bold text-white mb-1">
                  Stop simulation to add trip
                </h3>
                <p class="text-sm text-orange-100">
                  Click the stop button to add a new journey
                </p>
            </div>

          
              <!-- Finished Vehicle Alert - Now handled by JavaScript alert() -->
          </div>
            
              
        <!-- Loading indicator -->
        <div v-if="!networkData" class="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75">
          <div class="text-center">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p class="text-lg text-gray-600">Loading SUMO network...</p>
          </div>
          </div>

        <!-- Finished Vehicle Overlay - Arrow pointing to recent journeys -->
        <div v-if="showFinishedVehicleOverlay" class="absolute top-1/2 right-4 transform -translate-y-1/2 pointer-events-auto z-50 animate-fade-in">
          <div class="relative">
            <!-- Arrow pointing to recent journeys section -->
            <div class="bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg text-sm font-medium">
              Journey #{{ finishedVehicleMessage }} completed - check recent journeys
            </div>
            <!-- Arrow pointing left -->
            <div class="absolute left-0 top-1/2 transform -translate-x-1 -translate-y-1/2">
              <div class="w-0 h-0 border-t-4 border-b-4 border-r-4 border-transparent border-r-green-500"></div>
            </div>
            <!-- Close button -->
            <button 
              @click="closeFinishedVehicleOverlay"
              class="absolute -top-1 -right-1 bg-green-600 hover:bg-green-700 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold transition duration-200"
            >
              ×
            </button>
          </div>
        </div>
          </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Plot Modal -->
    <div v-if="showPlotModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" @click="closePlotModal">
      <div class="bg-slate-800 rounded-lg p-6 max-w-7xl w-full mx-4 max-h-[95vh] overflow-hidden" @click.stop>
        <!-- Modal Header -->
        <div class="flex justify-between items-center mb-2">
          <h3 class="text-2xl font-semibold text-slate-100">
            {{ currentPlotType === 'duration-vs-mae-scatter' ? 'Trip Duration vs MAE Scatter Plot' : 
               currentPlotType === 'distance-vs-mae-scatter' ? 'Trip Distance vs MAE Scatter Plot' : 
               currentPlotType === 'mae-by-time' ? 'MAE by Time of Day' : 
               currentPlotType === 'duration-histogram-short' ? 'Short Trips - Duration vs MAE Histogram' :
               currentPlotType === 'duration-histogram-medium' ? 'Medium Trips - Duration vs MAE Histogram' :
               currentPlotType === 'duration-histogram-long' ? 'Long Trips - Duration vs MAE Histogram' :
               currentPlotType === 'distance-histogram-short' ? 'Short Trips - Distance vs MAE Histogram' :
               currentPlotType === 'distance-histogram-medium' ? 'Medium Trips - Distance vs MAE Histogram' :
               currentPlotType === 'distance-histogram-long' ? 'Long Trips - Distance vs MAE Histogram' :
               currentPlotData?.title || 'Plot' }}
          </h3>
          <button 
            @click="closePlotModal"
            class="text-slate-400 hover:text-slate-200 transition-colors"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>
        
        <!-- Plot Content -->
        <div v-if="currentPlotData && (currentPlotType === 'duration-vs-mae-scatter' || currentPlotType === 'distance-vs-mae-scatter' || currentPlotType === 'mae-by-time' || currentPlotType === 'duration-histogram-short' || currentPlotType === 'duration-histogram-medium' || currentPlotType === 'duration-histogram-long' || currentPlotType === 'distance-histogram-short' || currentPlotType === 'distance-histogram-medium' || currentPlotType === 'distance-histogram-long')" class="space-y-2">
          <!-- Matplotlib Plot Image -->
          <div class="bg-slate-900 rounded-lg p-1">
            <div v-if="plotImage" class="w-full h-[calc(95vh-120px)] flex items-center justify-center">
              <img 
                :src="plotImage" 
                :alt="currentPlotType === 'duration-vs-mae-scatter' ? 'Trip Duration vs MAE Scatter Plot' : 
                       currentPlotType === 'distance-vs-mae-scatter' ? 'Trip Distance vs MAE Scatter Plot' : 
                       currentPlotType === 'mae-by-time' ? 'MAE by Time of Day Bar Chart' :
                       currentPlotType === 'duration-histogram-short' ? 'Short Trips - Duration vs MAE Histogram' :
                       currentPlotType === 'duration-histogram-medium' ? 'Medium Trips - Duration vs MAE Histogram' :
                       currentPlotType === 'duration-histogram-long' ? 'Long Trips - Duration vs MAE Histogram' :
                       currentPlotType === 'distance-histogram-short' ? 'Short Trips - Distance vs MAE Histogram' :
                       currentPlotType === 'distance-histogram-medium' ? 'Medium Trips - Distance vs MAE Histogram' :
                       currentPlotType === 'distance-histogram-long' ? 'Long Trips - Distance vs MAE Histogram' :
                       'Plot'"
                class="max-w-full max-h-full object-contain"
              />
            </div>
            
            <div v-else class="text-center text-slate-500 py-8">
              No plot image available
            </div>
          </div>
        </div>
        
        <!-- Placeholder for other plot types -->
        <div v-else class="text-center text-slate-400 py-8">
          Plot type not implemented yet
        </div>
      </div>
    </div>
    
    <!-- Seeding Indicator -->
    <div v-if="showSeedingIndicator" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div class="bg-slate-800 rounded-lg p-8 text-center">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <div class="text-slate-100 text-lg font-medium">Seeding Random Journeys...</div>
        <div class="text-slate-400 text-sm mt-2">Please wait while we generate test data</div>
      </div>
    </div>
  </div>
</template>

<script>
import apiService from '../services/api.js'
import jsPDF from 'jspdf'
// Chart.js will be imported dynamically

export default {
  name: 'DemoPage',
  components: {
  },
  data() {
    return {
      
      // Mouse position for proximity detection
      mousePosition: { x: 0, y: 0 },
      
      
      // Road switching debounce timer
      roadSwitchTimer: null,
      
      
      // Point placement
      startPoint: null,
      destinationPoint: null,
      
      // Journey state
      isJourneyRunning: false,
      
      // Simulation overlay
      showSimulationOverlay: false,
      
      // Expressway overlay
      showExpresswayOverlay: false,
      
      // Finished vehicle overlay
      showFinishedVehicleOverlay: false,
      finishedVehicleMessage: '',
      finishedVehicleTimer: null,
      shownFinishedVehicles: new Set(), // Track which vehicles we've already shown
      
      // Legend visibility
      showLegend: true,
      legendCollapsed: false,
      legendAutoFoldTimer: null,
      
      // Results tracking
      vehicleResults: [], // Array of vehicle journey results
      maxResults: 20, // Maximum number of results to keep
      totalJourneyCount: 0, // Total number of journeys in database
      
      // Statistics data
      journeyStatistics: {
        total_journeys: 0,
        average_duration: 0,
        average_distance: 0,
        mae: 0,
        rmse: 0,
        mape: 0,
        short_trips: {
          mae: 0,
          count: 0
        },
        medium_trips: {
          mae: 0,
          count: 0
        },
        long_trips: {
          mae: 0,
          count: 0
        },
        short_trips_distance: {
          mae: 0,
          count: 0
        },
        medium_trips_distance: {
          mae: 0,
          count: 0
        },
        long_trips_distance: {
          mae: 0,
          count: 0
        }
      },
      
      // Plot modal data
      showPlotModal: false,
      currentPlotData: null,
      currentPlotType: null,
      chartInstance: null,
      chartLoading: false,
      showFallbackTable: false,
      resizeHandler: null,
      plotImage: null,
      
      // Keyboard shortcuts
      showSeedingIndicator: false,
      
      // Route data
      calculatedRoute: null,
      
      // Invalid click feedback
      invalidClickMessage: '',
      
      // Animation
      animationInterval: null,

              // Network visualization
              networkData: null,
              networkBounds: null,
      svgViewBox: "0 0 1000 1000", // Default viewBox, will be calculated from network data
      
      // Simulation playback
      isSimulationPlaying: false,
      simulationStatus: {
        vehicles: 0,
        vehicles_in_route: 0,
        current_step: 0
      },
      
      // Vehicle management
      activeVehicles: [],
      vehicleUpdateInterval: null,
      
      // Simulation status
      simulationTime: 0,
      
      // Route data
      routePath: null,
      routeEdges: null,
      routeDistance: null,
      routeDuration: null,
      
      // Animation intervals
      simulationUpdateInterval: null,
      
      // PDF generation state
      isGeneratingPDF: false
    }
  },
  
  methods: {

    async loadNetworkData() {
      try {
        const response = await apiService.getNetworkData()
        this.networkData = response
        this.networkBounds = response.bounds
        
        
                // Calculate optimal viewBox with asymmetric padding
                // Network bounds: x: 0 to 18000, y: -6264.96 to 5000
                // Network width: 18000, height: 11264.96
                const paddingX = 0.02 // 2% padding on sides (restored)
                const paddingY = 0.01 // 1% padding on top and bottom (minimal)
                const width = this.networkBounds.max_x - this.networkBounds.min_x
                const height = this.networkBounds.max_y - this.networkBounds.min_y
                const paddingXValue = width * paddingX
                const paddingYValue = height * paddingY
                
                this.svgViewBox = `${this.networkBounds.min_x - paddingXValue} ${this.networkBounds.min_y - paddingYValue} ${width + 2 * paddingXValue} ${height + 2 * paddingYValue}`
        
        // Force reactive update
        this.$nextTick(() => {
          console.log('ViewBox updated:', this.svgViewBox)
        })
        console.log('Network data loaded:', {
          edges: this.networkData?.edges?.length || 0,
          junctions: this.networkData?.junctions?.length || 0,
          bounds: this.networkBounds,
          calculatedViewBox: {
            x: this.networkBounds.min_x - paddingXValue,
            y: this.networkBounds.min_y - paddingYValue,
            width: width + 2 * paddingXValue,
            height: height + 2 * paddingYValue
          }
        })
      } catch (error) {
        console.error('❌ Error loading network data:', error)
        this.networkData = null
      }
    },

    // Convert screen coordinates to network coordinates
    screenToNetwork(screenX, screenY) {
      // Validate input coordinates
      if (isNaN(screenX) || isNaN(screenY)) {
        console.warn('Invalid coordinates:', screenX, screenY)
        return { x: 0, y: 0 }
      }
      
      if (!this.networkBounds) {
        console.warn('No network bounds available')
        return { x: screenX, y: screenY }
      }
      
      const svgRect = this.$refs.networkSvg?.getBoundingClientRect()
      if (!svgRect) {
        console.warn('No SVG element found')
        return { x: screenX, y: screenY }
      }
      
      const scaleX = (this.networkBounds.max_x - this.networkBounds.min_x) / svgRect.width
      const scaleY = (this.networkBounds.max_y - this.networkBounds.min_y) / svgRect.height
      
      // screenX and screenY are already relative to the SVG element
      let networkX = this.networkBounds.min_x + screenX * scaleX
      let networkY = this.networkBounds.min_y + screenY * scaleY
      
      // Validate output coordinates
      if (isNaN(networkX) || isNaN(networkY)) {
        console.warn('Invalid network coordinates:', networkX, networkY, 'from', screenX, screenY)
        return { x: 0, y: 0 }
      }
      
      return { x: networkX, y: networkY }
    },

    // Convert network coordinates to screen coordinates
    networkToScreen(networkX, networkY) {
      if (!this.networkBounds) return { x: networkX, y: networkY }
      
      const svgRect = this.$refs.networkSvg?.getBoundingClientRect()
      if (!svgRect) return { x: networkX, y: networkY }
      
      const scaleX = svgRect.width / (this.networkBounds.max_x - this.networkBounds.min_x)
      const scaleY = svgRect.height / (this.networkBounds.max_y - this.networkBounds.min_y)
      
      return {
        x: (networkX - this.networkBounds.min_x) * scaleX,
        y: (networkY - this.networkBounds.min_y) * scaleY
      }
    },

    // Get junction position by ID
    getJunctionPosition(junctionId) {
      if (!this.networkData?.junctions) return null
      return this.networkData.junctions.find(j => j.id === junctionId)
    },
    
    isExpressEdge(edge) {
      // Check if an edge is an express edge based on naming patterns
      if (!edge || !edge.id) return false
      
      // Express edges are in the format -E0, -E1, -E2, etc. OR E0, E1, E2, etc.
      // Match the pattern -E followed by a number OR E followed by a number
      if (edge.id.match(/^-E\d+$/) || edge.id.match(/^E\d+$/)) {
        return true
      }
      
      return false
    },
    
    isEdgeClickable(edge) {
      // Expressways are not clickable when simulation is stopped
      if (!this.isSimulationPlaying && !this.isJourneyRunning && this.isExpressEdge(edge)) {
        return false
      }
      return true
    },
    
    getPathData(shapePoints) {
      if (!shapePoints || shapePoints.length < 2) return ''
      
      try {
        // Handle different data formats
        let points = []
        
        if (Array.isArray(shapePoints[0])) {
          // Format: [[x1, y1], [x2, y2], ...]
          points = shapePoints
        } else if (typeof shapePoints[0] === 'string') {
          // Check if it's coordinates or junction IDs
          if (shapePoints[0].includes(',')) {
            // Format: ["x1,y1", "x2,y2", ...] - parse comma-separated coordinate strings
            points = shapePoints.map(point => {
              const coords = point.split(',')
              return [parseFloat(coords[0]), parseFloat(coords[1])]
            })
          } else {
            // Format: ["JunctionID1", "JunctionID2", ...] - junction IDs, not coordinates
            console.warn('❌ Shape points appear to be junction IDs, not coordinates:', shapePoints)
            return ''  // Return empty string to trigger fallback to line rendering
          }
        } else {
          console.warn('❌ Unknown shape points format:', shapePoints)
          return ''
        }
        
        // Validate parsed points
        if (points.length < 2) {
          return ''
        }
        
        // Check if points are valid coordinates
        const firstPoint = points[0]
        if (!Array.isArray(firstPoint) || firstPoint.length < 2 || 
            isNaN(firstPoint[0]) || isNaN(firstPoint[1])) {
          return ''
        }
        
        // Create SVG path from points
        let pathData = `M ${points[0][0]} ${points[0][1]}`
        
        for (let i = 1; i < points.length; i++) {
          if (Array.isArray(points[i]) && points[i].length >= 2 && 
              !isNaN(points[i][0]) && !isNaN(points[i][1])) {
            pathData += ` L ${points[i][0]} ${points[i][1]}`
          }
        }
        
        return pathData
      } catch (error) {
        console.error('❌ Error creating path data:', error, 'Shape points:', shapePoints)
        return ''
      }
    },
    
    generateRoutePath(routeEdges) {
      if (!routeEdges || routeEdges.length < 2) return ''
      
      try {
        let pathData = ''
        
        // Start from S marker to beginning of second edge
        if (routeEdges.length >= 2) {
          const firstEdge = this.networkData?.edges?.find(e => e.id === routeEdges[0])
          const secondEdge = this.networkData?.edges?.find(e => e.id === routeEdges[1])
          
          if (firstEdge && secondEdge) {
            // Start at S marker (center of first edge)
            const startPoint = this.getEdgeCenterPoint(firstEdge)
            pathData = `M ${startPoint[0]} ${startPoint[1]}`
            
            // Go to the junction that connects to the second edge
            const firstEdgeEndJunction = this.getJunctionPosition(firstEdge.to_junction)
            if (firstEdgeEndJunction) {
              pathData += ` L ${firstEdgeEndJunction.x} ${firstEdgeEndJunction.y}`
            }
          }
        }
        
        // Paint intermediate edges (skip first and last)
        for (let i = 1; i < routeEdges.length - 1; i++) {
          const edgeId = routeEdges[i]
          const edge = this.networkData?.edges?.find(e => e.id === edgeId)
          if (!edge) continue
          
          // Paint the full edge shape
          if (edge.shape_points && edge.shape_points.length > 0) {
            for (const point of edge.shape_points) {
              pathData += ` L ${point[0]} ${point[1]}`
            }
          } else {
            // For edges without shape points, connect the junctions
            const startJunction = this.getJunctionPosition(edge.from_junction)
            const endJunction = this.getJunctionPosition(edge.to_junction)
            if (startJunction && endJunction) {
              pathData += ` L ${startJunction.x} ${startJunction.y}`
              pathData += ` L ${endJunction.x} ${endJunction.y}`
            }
          }
        }
        
        // Paint from end of second-to-last edge to D marker
        if (routeEdges.length >= 2) {
          const secondToLastEdge = this.networkData?.edges?.find(e => e.id === routeEdges[routeEdges.length - 2])
          const lastEdge = this.networkData?.edges?.find(e => e.id === routeEdges[routeEdges.length - 1])
          
          if (secondToLastEdge && lastEdge) {
            // Go to the junction that connects to the last edge
            const secondToLastEndJunction = this.getJunctionPosition(secondToLastEdge.to_junction)
            if (secondToLastEndJunction) {
              pathData += ` L ${secondToLastEndJunction.x} ${secondToLastEndJunction.y}`
            }
            
            // Go to D marker (center of last edge)
            const endPoint = this.getEdgeCenterPoint(lastEdge)
            pathData += ` L ${endPoint[0]} ${endPoint[1]}`
          }
        }
        
        console.log('🛣️ Generated route path:', pathData)
        return pathData
        
      } catch (error) {
        console.error('❌ Error generating route path:', error)
        return ''
      }
    },
    
    getEdgeCenterPoint(edge) {
      if (edge.shape_points && edge.shape_points.length > 0) {
        // For express edges with shape points, use the middle of the shape
        const midIndex = Math.floor(edge.shape_points.length / 2)
        if (edge.shape_points.length % 2 === 0 && midIndex > 0) {
          const point1 = edge.shape_points[midIndex - 1]
          const point2 = edge.shape_points[midIndex]
          return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]
        } else {
          return [edge.shape_points[midIndex][0], edge.shape_points[midIndex][1]]
        }
      } else {
        // For regular edges, use the midpoint between junctions
        const fromJunction = this.getJunctionPosition(edge.from_junction)
        const toJunction = this.getJunctionPosition(edge.to_junction)
        if (fromJunction && toJunction) {
          return [(fromJunction.x + toJunction.x) / 2, (fromJunction.y + toJunction.y) / 2]
        }
        return [0, 0] // Fallback
      }
    },
    
          // Vehicle color based on type and status
          getVehicleColor(vehicleType, status) {
            // White if stagnant
            if (status === 'stagnant') {
              return '#ffffff'
            }

            // Color based on vehicle type
            switch(vehicleType) {
              case 'passenger':
                return '#3b82f6'  // Blue
              case 'bus':
                return '#f59e0b'  // Orange
              case 'truck':
                return '#10b981'  // Green
              case 'user_defined':
                return '#fbbf24'  // Yellow
              default:
                return '#6b7280'  // Gray
            }
          },
    
          // Generate SVG path for a star shape
          getStarPath(cx, cy, radius) {
            const outerRadius = radius
            const innerRadius = radius * 0.4
            const points = 5
            let path = ''
            
            for (let i = 0; i < points * 2; i++) {
              const angle = (i * Math.PI) / points
              const r = i % 2 === 0 ? outerRadius : innerRadius
              const x = cx + r * Math.cos(angle - Math.PI / 2)
              const y = cy + r * Math.sin(angle - Math.PI / 2)
              
              if (i === 0) {
                path += `M ${x} ${y}`
              } else {
                path += ` L ${x} ${y}`
              }
            }
            path += ' Z'
            return path
          },
    
          // Vehicle stroke color
          getVehicleStroke(vehicleType, status) {
            // White stroke for stagnant vehicles
            if (status === 'stagnant') {
              return '#ffffff'
            }

            // Darker stroke for better visibility
            switch(vehicleType) {
              case 'passenger':
                return '#1d4ed8'  // Dark blue
              case 'bus':
                return '#d97706'  // Dark orange
              case 'truck':
                return '#059669'  // Dark green
              default:
                return '#374151'  // Dark gray
      }
    },

    // Convert lane shape coordinates to SVG path
    getLanePath(shape) {
      if (!shape || shape.length < 2) return ''
      
      // SUMO and SVG both use positive Y downward, so no flip needed
      let path = `M ${shape[0].x} ${shape[0].y}`
      for (let i = 1; i < shape.length; i++) {
        path += ` L ${shape[i].x} ${shape[i].y}`
      }
      return path
    },
    
    // Find the closest point on a line segment
    getClosestPointOnLine(px, py, x1, y1, x2, y2) {
      const A = px - x1
      const B = py - y1
      const C = x2 - x1
      const D = y2 - y1
      
      const dot = A * C + B * D
      const lenSq = C * C + D * D
      
      if (lenSq === 0) return { x: x1, y: y1 }
      
      let param = dot / lenSq
      
      // Clamp to line segment
      param = Math.max(0, Math.min(1, param))
      
      return {
        x: x1 + param * C,
        y: y1 + param * D
      }
    },
    
    // Find the closest point on a specific road
    findClosestPointOnRoad(clickX, clickY, road) {
      if (!road || !road.lanes) return null
      
      let closestPoint = null
      let minDistance = Infinity
      
      for (const lane of road.lanes) {
        if (!lane.shape || lane.shape.length < 2) continue
        
        // Check each segment of the lane
        for (let i = 0; i < lane.shape.length - 1; i++) {
          const point1 = lane.shape[i]
          const point2 = lane.shape[i + 1]
          
          // Find the closest point on this line segment
          const segmentPoint = this.getClosestPointOnLine(
            clickX, clickY,
            point1.x, point1.y,
            point2.x, point2.y
          )
          
          const distance = Math.sqrt(
            Math.pow(clickX - segmentPoint.x, 2) + 
            Math.pow(clickY - segmentPoint.y, 2)
          )
          
          if (distance < minDistance) {
            minDistance = distance
            closestPoint = {
              x: segmentPoint.x,
              y: segmentPoint.y,
              road: road,
              lane: lane
            }
          }
        }
      }
      
      return closestPoint
    },
    
    getInstructionText() {
      if (!this.startPoint) {
        return "Click on a road to set your starting location"
      } else if (!this.destinationPoint) {
        return "Click on a road to set your destination"
      } else if (!this.isSimulationPlaying) {
        return "Click 'Start Journey' to begin the simulation"
      } else {
        return "Vehicle is traveling to destination..."
      }
    },
    
    resetPoints() {
      // Reset both starting and destination points
      this.startPoint = null
      this.destinationPoint = null
      this.routePath = null
      this.routeEdges = null
      this.routeDistance = null
      this.routeDuration = null
      this.isJourneyRunning = false
      console.log('🔄 Points reset - ready to select new starting point')
    },
    
    
    getMainButtonText() {
      if (this.isSimulationPlaying) {
        return '⏹️ Stop Simulation'
      } else if (this.isJourneyRunning) {
        return '⏹️ Stop Simulation'
      } else if (this.startPoint && this.destinationPoint) {
        return '🚀 Start Journey'
      } else {
        return '▶️ Play Simulation'
      }
    },
    
    getMainButtonClass() {
      if (this.isSimulationPlaying) {
        return 'bg-red-600 hover:bg-red-700'
      } else if (this.isJourneyRunning) {
        return 'bg-red-600 hover:bg-red-700'
      } else if (this.startPoint && this.destinationPoint) {
        return 'bg-green-600 hover:bg-green-700'
      } else {
        return 'bg-blue-600 hover:bg-blue-700'
      }
    },
    
    async handleMainButtonClick() {
      if (this.isSimulationPlaying) {
        this.stopSimulationPlayback()
        // Reset points when stopping main simulation
        this.startPoint = null
        this.destinationPoint = null
        this.routePath = null
        this.routeDistance = null
        this.routeDuration = null
        this.isJourneyRunning = false
        console.log('🔄 Points reset after stopping main simulation')
      } else if (this.isJourneyRunning) {
        this.stopJourney()
      } else if (this.startPoint && this.destinationPoint) {
        // Ensure route is calculated before starting journey
        if (!this.routeEdges || !Array.isArray(this.routeEdges)) {
          console.log('🔄 Route not calculated yet, calculating now...')
          await this.calculateAndDisplayRoute()
          
          // Check if route calculation was successful
          if (!this.routeEdges || !Array.isArray(this.routeEdges)) {
            console.error('❌ Route calculation failed')
            alert('Failed to calculate route. Please try selecting different points.')
            return
          }
        }
        this.startJourney()
      } else {
        this.startSimulationPlayback()
      }
    },
    
    async calculateAndDisplayRoute() {
      if (!this.startPoint || !this.destinationPoint) return
      
      console.log('🛣️ Calculating route from', this.startPoint.id, 'to', this.destinationPoint.id)
      
      try {
        // Calculate route between edges
        const routeResponse = await apiService.calculateRouteByEdges(
          this.startPoint.id, 
          this.destinationPoint.id
        )
        
        if (routeResponse.error) {
          console.error('❌ Route calculation failed:', routeResponse.error)
          alert(`Route calculation failed: ${routeResponse.error}`)
          return
        }
        
        console.log('✅ Route calculated:', routeResponse)
        
        // Store the route for display
        this.routeEdges = routeResponse.edges
        this.routeDistance = routeResponse.distance
        this.routeDuration = routeResponse.duration
        
        // Generate SVG path for the route
        this.routePath = this.generateRoutePath(this.routeEdges)
        
        console.log('🟢 Route displayed on map - Edges:', this.routeEdges.join(', '))
        
      } catch (error) {
        console.error('❌ Error calculating route:', error)
        alert('Failed to calculate route. Please try again.')
      }
    },

    async startJourney() {
      if (!this.startPoint || !this.destinationPoint) return
      
      console.log('🚀 Starting journey from', this.startPoint, 'to', this.destinationPoint)
      
      try {
        
        if (!this.routeEdges || !Array.isArray(this.routeEdges)) {
          console.error('❌ Route edges not available or not an array')
          alert('Route not calculated. Please select start and destination points first.')
          return
        }
        
        // Convert Proxy array to plain JavaScript array
        const routeEdgesArray = Array.from(this.routeEdges)
        
        
        // Call the regular API to add vehicle to simulation
        const response = await apiService.startJourney(
          this.startPoint.id,
          this.destinationPoint.id,
          routeEdgesArray
        )
        
        console.log('✅ Journey started:', response)
        
        // Add vehicle result to local tracking
        this.addVehicleResult(
          response.vehicle_id, 
          response.start_time, 
          response.start_time_string, 
          response.distance, 
          response.predicted_eta,
          this.startPoint.id,
          this.destinationPoint.id,
          routeEdgesArray
        )
        
        // Start simulation with additional vehicle
        this.startSimulationPlayback()
        this.isJourneyRunning = true
        
        console.log('🚗 Vehicle added to simulation with route')
      } catch (error) {
        console.error('❌ Error starting journey:', error)
        alert('Failed to start journey. Please try again.')
      }
    },
    
    stopJourney() {
      console.log('⏹️ Stopping journey')
      
      // Stop the simulation
      this.stopSimulationPlayback()
      this.isJourneyRunning = false
      
      // Reset the points
      this.startPoint = null
      this.destinationPoint = null
      this.routePath = null
      this.routeEdges = null
      this.routeDistance = null
      this.routeDuration = null
      
      // TODO: Remove vehicle from simulation
      console.log('🚗 Vehicle removed from simulation')
      console.log('🔄 Points reset after journey stop')
    },
    
    handleMapClick(event) {
      // Only show overlay if simulation is running and clicking on empty SVG area
      if ((this.isSimulationPlaying || this.isJourneyRunning) && event.target.tagName === 'svg') {
        this.showSimulationOverlay = true
        console.log('🚗 Simulation running - showing overlay on map click')
      }
    },
    
    closeSimulationOverlay() {
      this.showSimulationOverlay = false
      console.log('✅ Simulation overlay closed')
    },
    
    closeFinishedVehicleOverlay() {
      this.showFinishedVehicleOverlay = false
      this.finishedVehicleMessage = ''
      
      // Clear the auto-close timer if it exists
      if (this.finishedVehicleTimer) {
        clearTimeout(this.finishedVehicleTimer)
        this.finishedVehicleTimer = null
      }
    },
    
    handleEdgeClick(edge, event) {
      // Only allow clicking when simulation is stopped
      if (this.isSimulationPlaying || this.isJourneyRunning) {
        event.stopPropagation()
        this.showSimulationOverlay = true
        return
      }
      
      // Check if it's an expressway when simulation is stopped
      if (this.isExpressEdge(edge)) {
        event.stopPropagation()
        this.showExpresswayOverlay = true
        
        // Hide overlay after 5 seconds
        setTimeout(() => {
          this.showExpresswayOverlay = false
        }, 5000)
        return
      }
      
      // Calculate the exact middle position of the edge
      let x, y
      
      if (edge.shape_points && edge.shape_points.length > 0) {
        // For express edges with shape points, calculate the exact middle point
        const totalLength = edge.shape_points.length
        const midIndex = Math.floor(totalLength / 2)
        
        // If we have an even number of points, interpolate between the two middle points
        if (totalLength % 2 === 0 && midIndex > 0) {
          const point1 = edge.shape_points[midIndex - 1]
          const point2 = edge.shape_points[midIndex]
          x = (point1[0] + point2[0]) / 2
          y = (point1[1] + point2[1]) / 2
        } else {
          // Use the exact middle point
          x = edge.shape_points[midIndex][0]
          y = edge.shape_points[midIndex][1]
        }
      } else {
        // For regular edges, use the exact midpoint between junctions
        const fromJunction = this.getJunctionPosition(edge.from_junction)
        const toJunction = this.getJunctionPosition(edge.to_junction)
        if (fromJunction && toJunction) {
          x = (fromJunction.x + toJunction.x) / 2
          y = (fromJunction.y + toJunction.y) / 2
        } else {
          // Fallback to center of view
          x = 0
          y = 0
        }
      }
      
      if (!this.startPoint) {
        // Set starting point
        this.startPoint = { id: edge.id, x: x, y: y }
        this.isJourneyRunning = false // Reset journey state when new starting point is set
        console.log('📍 Starting point set:', this.startPoint)
      } else if (!this.destinationPoint) {
        // Set destination point
        this.destinationPoint = { id: edge.id, x: x, y: y }
        this.isJourneyRunning = false // Reset journey state when new destination point is set
        console.log('🎯 Destination point set:', this.destinationPoint)
        
        // Calculate and display route immediately when destination is selected
        this.calculateAndDisplayRoute()
      }
      // If both points are already set, do nothing (user should use reset button)
    },
    
    getRoadClass(edge) {
      let baseClass = "road-path"
      
      // Check if it's an expressway (any pattern starting with E)
      if (edge.id && (edge.id.startsWith('-E') || edge.id.startsWith('E'))) {
        baseClass += " expressway"
      }
      
      // Add clickable class when simulation is stopped
      if (!this.isSimulationPlaying) {
        baseClass += " clickable"
      }
      
      return baseClass
    },
    
    getRoadStrokeColor(edge) {
      // Expressways are blue, regular roads are dark gray for better visibility
      if (edge.id && (edge.id.startsWith('-E') || edge.id.startsWith('E'))) {
        return "#1e40af"  // Darker blue for expressways
      }
      return "#374151"  // Dark gray for regular roads - more visible than black
    },
    
    getRoadStrokeWidth(edge) {
      // All roads have thick strokes
      if (edge.id && (edge.id.startsWith('-E') || edge.id.startsWith('E'))) {
        return "17"  // Expressways
      }
      return "15"  // Regular roads
    },
    
    getRoadOpacity(edge) {
      // Expressways are dimmed, regular roads are full opacity
      if (edge.id && (edge.id.startsWith('-E') || edge.id.startsWith('E'))) {
        return "0.6"
      }
      return "1.0"
    },
    
    isExpressway(edge) {
      return edge.id && (edge.id.startsWith('-E') || edge.id.startsWith('E'))
    },
    
    isRouteEdge(edge) {
      // Check if this edge is part of the calculated route (only intermediate edges)
      if (!this.routeEdges || !Array.isArray(this.routeEdges)) return false
      
      const edgeIndex = this.routeEdges.indexOf(edge.id)
      // Only highlight intermediate edges (skip first and last)
      return edgeIndex > 0 && edgeIndex < this.routeEdges.length - 1
    },
    
    getRouteStrokeColor(edge) {
      // Return bright fluorescent green for route edges
      if (this.isRouteEdge(edge)) {
        return "#00ff00"  // Bright green
      }
      
      // Expressways are blue, regular roads are dark gray for better visibility
      if (this.isExpressway(edge)) {
        return "#1e40af"  // Darker blue for expressways
      }
      return "#374151"  // Dark gray for regular roads - more visible than black
    },
    
    getRouteStrokeWidth(edge) {
      // Make route edges much thicker and more visible
      if (this.isRouteEdge(edge)) {
        return "40"  // Extra thick for route
      }
      
      return this.getRoadStrokeWidth(edge)
    },
    
    
    showInvalidClickFeedback() {
      // Show a temporary message that the click was not on a road
      this.invalidClickMessage = "Please click on a road, not empty space"
      setTimeout(() => {
        this.invalidClickMessage = ""
      }, 2000)
    },
    
    showExpresswayNotAllowedFeedback() {
      // Show a temporary message that expressways are not allowed
      this.invalidClickMessage = "Expressways are not allowed for point placement. Please choose a regular road."
      setTimeout(() => {
        this.invalidClickMessage = ""
      }, 3000)
    },
    
    
    handleMouseMove(event) {
      // Track mouse position for proximity detection
      const svg = this.$refs.networkSvg
      if (!svg) return

      const rect = svg.getBoundingClientRect()
      this.mousePosition = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      }
    },
    
    // Touch event handlers for mobile devices
    handleTouchStart(event) {
      event.preventDefault()
      if (event.touches.length === 1) {
        const touch = event.touches[0]
        const svg = this.$refs.networkSvg
        if (!svg) return

        const rect = svg.getBoundingClientRect()
        this.mousePosition = {
          x: touch.clientX - rect.left,
          y: touch.clientY - rect.top
        }
      }
    },
    
    handleTouchMove(event) {
      event.preventDefault()
      if (event.touches.length === 1) {
        const touch = event.touches[0]
        const svg = this.$refs.networkSvg
        if (!svg) return

        const rect = svg.getBoundingClientRect()
        this.mousePosition = {
          x: touch.clientX - rect.left,
          y: touch.clientY - rect.top
        }
      }
    },
    
    handleTouchEnd(event) {
      event.preventDefault()
      if (event.changedTouches.length === 1) {
        const touch = event.changedTouches[0]
        const svg = this.$refs.networkSvg
        if (!svg) return

        const rect = svg.getBoundingClientRect()
        const clickEvent = {
          clientX: touch.clientX,
          clientY: touch.clientY,
          target: event.target
        }
        
        // Simulate a click event for touch
        this.handleMapClick(clickEvent)
      }
    },
    
    
    
    async calculateRoute() {
      if (!this.startPoint || !this.destinationPoint) return
      
      try {
        console.log('🛣️ Calculating route...', {
          startEdge: this.startPoint.road.id,
          endEdge: this.destinationPoint.road.id
        })
        
        const routeData = await apiService.calculateRouteByEdges(
          this.startPoint.road.id,
          this.destinationPoint.road.id
        )
        
        if (routeData.success) {
          this.calculatedRoute = routeData
          console.log('✅ Route calculated:', routeData)
          console.log('🟠 Route edges:', routeData.route_edges)
          console.log('🟠 Calculated route set:', this.calculatedRoute)
        } else {
          console.error('❌ Route calculation failed:', routeData.error)
        }
      } catch (error) {
        console.error('❌ Error calculating route:', error)
      }
    },
    
    async startSimulation() {
      if (!this.startPoint || !this.destinationPoint) return
      
      try {
        // Call the real API to start journey
        const response = await apiService.startJourney(
          { x: this.startPoint.x, y: this.startPoint.y },
          { x: this.destinationPoint.x, y: this.destinationPoint.y }
        )
        
        this.predictedETA = response.predicted_eta
        this.currentPrediction = this.predictedETA
        this.currentVehicleId = response.vehicle_id
        this.simulationStartTime = Date.now()
        
        // Create route path (simplified straight line for demo)
        const startX = this.startPoint.x
        const startY = this.startPoint.y
        const endX = this.destinationPoint.x
        const endY = this.destinationPoint.y
        
        this.routePath = `M ${startX} ${startY} L ${endX} ${endY}`
      } catch (error) {
        console.error('Error starting journey:', error)
        // Fallback to simulated data
        this.predictedETA = Math.floor(Math.random() * 20) + 5
        this.currentPrediction = this.predictedETA
        this.currentVehicleId = `vehicle_${Date.now()}`
        this.simulationStartTime = Date.now()
        
        const startX = this.startPoint.x
        const startY = this.startPoint.y
        const endX = this.destinationPoint.x
        const endY = this.destinationPoint.y
        
        this.routePath = `M ${startX} ${startY} L ${endX} ${endY}`
      }
    },
    
    startSimulation() {
      if (!this.startPoint) return
      
      // Reset the source point as requested
      this.startPoint = null
      this.destinationPoint = null
      this.routePath = null
      this.isJourneyRunning = false
      
      // Start the simulation
      this.startSimulationPlayback()
      console.log('▶️ Simulation started - source point reset')
    },
    
    animateVehicle() {
      const startX = this.startPoint.x
      const startY = this.startPoint.y
      const endX = this.destinationPoint.x
      const endY = this.destinationPoint.y
      
      const totalDistance = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2))
      const duration = this.predictedETA * 1000 // Convert to milliseconds
      const steps = 60 // 60 steps for smooth animation
      const stepDuration = duration / steps
      
      let currentStep = 0
      
      this.animationInterval = setInterval(() => {
        currentStep++
        const progress = currentStep / steps
        
        if (progress >= 1) {
          // Journey complete
          this.completeJourney()
          return
        }
        
        // Update vehicle position
        this.vehiclePosition = {
          x: startX + (endX - startX) * progress,
          y: startY + (endY - startY) * progress
        }
      }, stepDuration)
    },
    
    async completeJourney() {
      this.isSimulationPlaying = false
      this.actualDuration = Math.floor((Date.now() - this.simulationStartTime) / 1000 / 60) // Convert to minutes
      
      try {
        // Call the real API to complete journey
        if (this.currentVehicleId) {
          await apiService.completeJourney(this.currentVehicleId, this.actualDuration * 60) // Convert to seconds
        }
        
        // Update vehicle result with completion data
        const endTime = this.simulationTime
        const actualDuration = this.actualDuration * 60 // Convert to seconds
        const predictedDuration = this.predictedETA * 60 // Convert to seconds
        
        this.updateVehicleResult(this.currentVehicleId, endTime, actualDuration, predictedDuration)
        
      } catch (error) {
        console.error('Error completing journey:', error)
        // Fallback to local calculation
        this.calculateLocalResult()
      }
      
      // Show journey summary instead of immediately resetting
      this.showJourneySummary = true
      
      if (this.animationInterval) {
        clearInterval(this.animationInterval)
        this.animationInterval = null
      }
    },
    
    calculateLocalResult() {
      // Calculate accuracy
      const error = Math.abs(this.predictedETA - this.actualDuration)
      const accuracy = Math.max(0, 100 - (error / this.predictedETA) * 100)
      
      // Calculate distance (simplified)
      const distance = Math.sqrt(
        Math.pow(this.destinationPoint.x - this.startPoint.x, 2) + 
        Math.pow(this.destinationPoint.y - this.startPoint.y, 2)
      ) / 10 // Scale factor for demo
      
      // Create result record
      const result = {
        id: Date.now(),
        timestamp: new Date().toLocaleString(),
        predictedETA: this.predictedETA,
        actualDuration: this.actualDuration,
        distance: distance.toFixed(1),
        accuracy: Math.round(accuracy)
      }
      
      // Add to results
      this.recentResults.unshift(result)
      if (this.recentResults.length > 10) {
        this.recentResults = this.recentResults.slice(0, 10)
      }
      
      // Update statistics
      this.updateStatistics(result)
    },
    
    updateStatistics(newResult) {
      this.statistics.totalRuns++
      this.statistics.totalDistance += parseFloat(newResult.distance)
      
      // Calculate average accuracy
      const totalAccuracy = this.recentResults.reduce((sum, result) => sum + result.accuracy, 0)
      this.statistics.avgAccuracy = Math.round(totalAccuracy / this.recentResults.length)
      
      // Calculate average error
      const totalError = this.recentResults.reduce((sum, result) => {
        return sum + Math.abs(result.predictedETA - result.actualDuration)
      }, 0)
      this.statistics.avgError = Math.round(totalError / this.recentResults.length)
    },
    
    
    resetSimulation() {
      this.isSimulationPlaying = false
      this.vehiclePosition = null
      this.routePath = null
      this.routeEdges = null
      this.routeDistance = null
      this.routeDuration = null
      this.currentPrediction = null
      this.predictedETA = null
      this.actualDuration = null
      this.isJourneyRunning = false
      
      // Also reset the points so user can start fresh
      this.startPoint = null
      this.destinationPoint = null
      
      if (this.animationInterval) {
        clearInterval(this.animationInterval)
        this.animationInterval = null
      }
    },
    
    getAccuracyClass(accuracy) {
      if (accuracy >= 90) return 'bg-green-100 text-green-800'
      if (accuracy >= 70) return 'bg-yellow-100 text-yellow-800'
      return 'bg-red-100 text-red-800'
    },
    
    calculateAccuracy() {
      if (!this.predictedETA || !this.actualDuration) return 0
      const error = Math.abs(this.predictedETA - this.actualDuration)
      const accuracy = Math.max(0, 100 - (error / this.predictedETA) * 100)
      return Math.round(accuracy)
    },
    
    getAccuracySummaryClass() {
      const accuracy = this.calculateAccuracy()
      if (accuracy >= 90) return 'bg-green-50'
      if (accuracy >= 70) return 'bg-yellow-50'
      return 'bg-red-50'
    },
    
    getAccuracySummaryTextClass() {
      const accuracy = this.calculateAccuracy()
      if (accuracy >= 90) return 'text-green-600'
      if (accuracy >= 70) return 'text-yellow-600'
      return 'text-red-600'
    },
    
    closeJourneySummary() {
      this.showJourneySummary = false
      // Reset everything for next journey
      this.resetSimulation()
    },
    
    
    
    
    // Real-time vehicle methods
    
    // Simulation playback methods
    async loadSimulationStatus() {
      try {
        const response = await apiService.getSimulationStatus()
        this.simulationStatus = response
        this.isSimulationPlaying = response.is_running || false
        
        // Convert simulation time string to seconds
        if (response.simulation_time && typeof response.simulation_time === 'string') {
          const parts = response.simulation_time.split(':')
          if (parts.length >= 3) {
            const hours = parseInt(parts[0]) || 0
            const minutes = parseInt(parts[1]) || 0
            const secs = parseInt(parts[2]) || 0
            this.simulationTime = hours * 3600 + minutes * 60 + secs
          } else {
            this.simulationTime = 0
          }
        } else {
          this.simulationTime = response.simulation_time || 0
        }
        
      } catch (error) {
        console.error('Error loading simulation status:', error)
      }
    },

    async loadResults() {
      try {
        // Load results from API
        const response = await apiService.getSimulationResults(this.maxResults)
        this.vehicleResults = response.results || []
        console.log('📊 Results loaded:', this.vehicleResults.length)
      } catch (error) {
        console.error('❌ Failed to load results:', error)
        this.vehicleResults = []
      }
    },

    async loadRecentJourneysFromDB() {
      try {
        console.log('📊 Loading recent journeys from database...')
        const response = await apiService.getRecentJourneys(20)
        
        if (response.success && response.journeys) {
          // Store total journey count
          this.totalJourneyCount = response.total_count || 0
          
          // Transform database format to frontend format
          this.vehicleResults = response.journeys.map(journey => ({
            vehicle_id: journey.vehicle_id,
            start_time: journey.start_time,
            start_time_string: journey.start_time_string,
            end_time: journey.end_time,
            end_time_string: journey.end_time_string,
            distance: journey.distance,
            predicted_eta: journey.predicted_eta,
            actual_duration: journey.actual_duration,
            absolute_error: journey.absolute_error,
            accuracy: journey.accuracy,
            status: journey.status,
            start_edge: journey.start_edge,
            end_edge: journey.end_edge,
            route_edges: journey.route_edges
          }))
          
          console.log('✅ Recent journeys loaded from database:', this.vehicleResults.length, 'Total count:', this.totalJourneyCount)
        } else {
          console.log('📊 No journeys found in database')
          this.vehicleResults = []
        }
      } catch (error) {
        console.error('❌ Failed to load recent journeys from database:', error)
        // Fallback to empty array on error
        this.vehicleResults = []
      }
    },

    async loadJourneyStatistics() {
      try {
        console.log('📊 Loading journey statistics...')
        const response = await apiService.getJourneyStatistics()
        
        if (response.success && response.statistics) {
          this.journeyStatistics = response.statistics
          console.log('✅ Journey statistics loaded:', this.journeyStatistics)
        } else {
          console.log('📊 No statistics available')
          this.journeyStatistics = {
            total_journeys: 0,
            average_duration: 0,
            average_distance: 0,
            mae: 0,
            rmse: 0,
            mape: 0,
            short_trips: { mae: 0, count: 0 },
            medium_trips: { mae: 0, count: 0 },
            long_trips: { mae: 0, count: 0 },
            short_trips_distance: { mae: 0, count: 0 },
            medium_trips_distance: { mae: 0, count: 0 },
            long_trips_distance: { mae: 0, count: 0 }
          }
        }
      } catch (error) {
        console.error('❌ Failed to load journey statistics:', error)
        // Keep default values on error
        this.journeyStatistics = {
          total_journeys: 0,
          average_duration: 0,
          average_distance: 0,
          mae: 0,
          rmse: 0,
          mape: 0,
          short_trips: { mae: 0, count: 0 },
          medium_trips: { mae: 0, count: 0 },
          long_trips: { mae: 0, count: 0 },
          short_trips_distance: { mae: 0, count: 0 },
          medium_trips_distance: { mae: 0, count: 0 },
          long_trips_distance: { mae: 0, count: 0 }
        }
      }
    },
    
    addVehicleResult(vehicleId, startTime, startTimeString, distance, predictedEta, startEdge = null, endEdge = null, routeEdges = null) {
      const result = {
        vehicle_id: vehicleId,
        start_time: startTime,
        start_time_string: startTimeString,
        distance: distance,
        predicted_eta: predictedEta,
        start_edge: startEdge,
        end_edge: endEdge,
        route_edges: routeEdges,
        status: 'running',
        end_time: null,
        end_time_string: null,
        actual_duration: null,
        absolute_error: null,
        accuracy: null
      }
      
      // Add to beginning of array (most recent first)
      this.vehicleResults.unshift(result)
      
      // Keep only maxResults
      if (this.vehicleResults.length > this.maxResults) {
        this.vehicleResults = this.vehicleResults.slice(0, this.maxResults)
      }
      
      console.log('📊 Vehicle result added:', result)
    },
    
    
    // Vehicle management methods
    async loadActiveVehicles() {
      try {
        const response = await apiService.getActiveVehicles()
        this.activeVehicles = response.vehicles || []
      } catch (error) {
        console.error('Error loading active vehicles:', error)
        this.activeVehicles = []
      }
    },
    
    startVehicleUpdates() {
      // Load vehicles immediately
      this.loadActiveVehicles()
      
      // Set up interval for real-time updates
      this.vehicleUpdateInterval = setInterval(() => {
        if (this.isSimulationPlaying) {
          this.loadActiveVehicles()
        }
        // Always check for finished vehicles regardless of simulation state
        this.checkFinishedVehicles()
      }, 1000) // Update every second
    },
    
    stopVehicleUpdates() {
      if (this.vehicleUpdateInterval) {
        clearInterval(this.vehicleUpdateInterval)
        this.vehicleUpdateInterval = null
        console.log('🛑 Stopped vehicle updates')
      }
    },
    
    startLegendAutoFold() {
      // Clear any existing timer
      if (this.legendAutoFoldTimer) {
        clearTimeout(this.legendAutoFoldTimer)
      }
      
      // Set timer to collapse legend after 10 seconds
      this.legendAutoFoldTimer = setTimeout(() => {
        this.legendCollapsed = true
        console.log('🕐 Legend auto-folded after 10 seconds')
      }, 10000)
    },
    
    resetLegendAutoFold() {
      // Clear existing timer
      if (this.legendAutoFoldTimer) {
        clearTimeout(this.legendAutoFoldTimer)
      }
      
      // Start new timer
      this.startLegendAutoFold()
    },
    
    toggleLegend() {
      this.legendCollapsed = !this.legendCollapsed
      
      // Reset auto-fold timer when user manually toggles
      this.resetLegendAutoFold()
    },
    

    clearAllResults() {
      this.vehicleResults = []
      console.log('🗑️ All results cleared')
    },

    async deleteLastJourney() {
      try {
        console.log('🗑️ Deleting last journey from database...')
        const response = await apiService.deleteLastJourney()
        
        if (response.success) {
          console.log('✅ Last journey deleted:', response.deleted_journey_number)
          // Refresh the recent journeys from database
          await this.loadRecentJourneysFromDB()
          // Refresh statistics after deleting journey
          await this.loadJourneyStatistics()
        } else {
          console.log('ℹ️ No journeys found to delete')
        }
      } catch (error) {
        console.error('❌ Error deleting last journey:', error)
      }
    },

    async deleteAllJourneys() {
      try {
        console.log('🗑️ Deleting all journeys from database...')
        const response = await apiService.deleteAllJourneys()
        
        if (response.success) {
          console.log('✅ All journeys deleted:', response.deleted_count)
          // Clear local results and refresh from database
          this.vehicleResults = []
          await this.loadRecentJourneysFromDB()
          // Refresh statistics after deleting all journeys
          await this.loadJourneyStatistics()
        }
      } catch (error) {
        console.error('❌ Error deleting all journeys:', error)
      }
    },

    addHiddenCommandListener() {
      // Hidden commands: C+A+L (delete last), C+A+D (delete all)
      console.log('🔧 Hidden commands available:')
      console.log('  - C+A+L: Delete last journey')
      console.log('  - C+A+D: Delete all journeys')
      
      this.hiddenCommandHandler = (event) => {
        // Check for C+A combination
        if (event.key.toLowerCase() === 'c') {
          event.preventDefault()
          // Wait for the next key press to determine the action
          const handleNextKey = (nextEvent) => {
            if (nextEvent.key.toLowerCase() === 'a') {
              nextEvent.preventDefault()
              // Wait for the final key press
              const handleFinalKey = (finalEvent) => {
                if (finalEvent.key.toLowerCase() === 'l') {
                  finalEvent.preventDefault()
                  console.log('🗑️ Hidden command triggered: Delete last journey')
                  this.deleteLastJourney()
                } else if (finalEvent.key.toLowerCase() === 'd') {
                  finalEvent.preventDefault()
                  console.log('🗑️ Hidden command triggered: Delete all journeys')
                  this.deleteAllJourneys()
                }
                document.removeEventListener('keydown', handleFinalKey)
              }
              document.addEventListener('keydown', handleFinalKey)
            }
            document.removeEventListener('keydown', handleNextKey)
          }
          document.addEventListener('keydown', handleNextKey)
        }
      }
      document.addEventListener('keydown', this.hiddenCommandHandler)
    },
    
    scrollToTop() {
      const resultsContainer = this.$el.querySelector('.overflow-y-auto')
      if (resultsContainer) {
        resultsContainer.scrollTo({ top: 0, behavior: 'smooth' })
      }
    },
    
    async updateVehicleResult(vehicleId, endTime, actualDuration, predictedDuration) {
      // Update local vehicle result
      
      const result = this.vehicleResults.find(r => r.vehicle_id === vehicleId)
      if (result) {
        result.status = 'finished'
        result.end_time = endTime
        result.end_time_string = this.formatTime(endTime)
        result.actual_duration = actualDuration
        
        // Calculate error and accuracy
        const absoluteError = Math.abs(predictedDuration - actualDuration)
        const accuracy = actualDuration > 0 ? Math.max(0, 100 - (absoluteError / actualDuration) * 100) : 0
        
        result.absolute_error = absoluteError
        result.accuracy = accuracy
        
        console.log('📊 Vehicle result updated:', result)
        
        // Save journey to database only after results are displayed
        try {
          await this.saveJourneyToDatabase(result)
          // Refresh statistics after saving journey
          await this.loadJourneyStatistics()
        } catch (error) {
          console.error('❌ Error saving journey to database:', error)
        }
      } else {
        console.warn('⚠️ Vehicle result not found for vehicle ID:', vehicleId)
        console.log('📊 Available vehicle results:', this.vehicleResults)
      }
    },

    async saveJourneyToDatabase(journeyResult) {
      try {
        // Prepare journey data for database
        const journeyData = {
          vehicle_id: journeyResult.vehicle_id,
          start_edge: journeyResult.start_edge || 'unknown',
          end_edge: journeyResult.end_edge || 'unknown',
          route_edges: journeyResult.route_edges || [],
          start_time: journeyResult.start_time,
          start_time_string: journeyResult.start_time_string,
          end_time: journeyResult.end_time,
          end_time_string: journeyResult.end_time_string,
          distance: journeyResult.distance,
          predicted_eta: journeyResult.predicted_eta,
          actual_duration: journeyResult.actual_duration,
          absolute_error: journeyResult.absolute_error,
          accuracy: journeyResult.accuracy,
          status: journeyResult.status
        }

        console.log('💾 Saving journey to database:', journeyData)
        
        const response = await apiService.saveJourney(journeyData)
        console.log('✅ Journey saved to database:', response)
        
        // Update total journey count
        if (response.success && response.journey_number) {
          this.totalJourneyCount = Math.max(this.totalJourneyCount, response.journey_number)
        }
        
        return response
      } catch (error) {
        console.error('❌ Error saving journey to database:', error)
        throw error
      }
    },
    
    async checkFinishedVehicles() {
      try {
        const response = await apiService.getFinishedVehicles()
        const finishedVehicles = response.finished_vehicles || []
        
        
        for (const vehicle of finishedVehicles) {
          
          // Skip if we've already shown this vehicle
          if (this.shownFinishedVehicles.has(vehicle.id)) {
            continue
          }
          
          const duration = vehicle.end_time - vehicle.start_time
          let message = `Vehicle ${vehicle.id} finished journey in ${this.formatTime(duration)} (${duration.toFixed(1)}s)`
          
          // Add prediction information if available
          if (vehicle.prediction) {
            const prediction = vehicle.prediction
            const predictedDuration = prediction.predicted_travel_time
            const actualDuration = duration
            
            // Update vehicle result with completion data
            this.updateVehicleResult(vehicle.id, vehicle.end_time, actualDuration, predictedDuration)
            
            // Calculate accuracy, handling division by zero
            let accuracyText = 'N/A'
            if (actualDuration > 0) {
              const accuracy = 1.0 - Math.abs(predictedDuration - actualDuration) / actualDuration
              accuracyText = `${(accuracy * 100).toFixed(1)}%`
            }
            
            message += `\n\n🎯 ETA Prediction Results:`
            message += `\nPredicted: ${this.formatTime(predictedDuration)} (${predictedDuration.toFixed(1)}s)`
            message += `\nActual: ${this.formatTime(actualDuration)} (${actualDuration.toFixed(1)}s)`
            message += `\nAccuracy: ${accuracyText}`
          } else {
            // Update vehicle result with completion data
            this.updateVehicleResult(vehicle.id, vehicle.end_time, duration, 0)
          }
          
          // Mark vehicle as shown
          this.shownFinishedVehicles.add(vehicle.id)
          console.log('✅ Marked vehicle as shown:', vehicle.id)
          
            // Find the journey number for this vehicle
            const result = this.vehicleResults.find(r => r.vehicle_id === vehicle.id)
            const journeyNumber = result ? (this.totalJourneyCount - this.vehicleResults.indexOf(result)) : '?'
          
          // Show overlay for finished vehicle
          this.showFinishedVehicleOverlay = true
          this.finishedVehicleMessage = journeyNumber
          console.log('🎉 Showing finished vehicle overlay for journey #', journeyNumber)
          
          // Auto-close after 20 seconds
          this.finishedVehicleTimer = setTimeout(() => {
            this.closeFinishedVehicleOverlay()
          }, 20000)
        }
      } catch (error) {
        console.error('Error checking finished vehicles:', error)
      }
    },
    
    async openPlotWindow(plotType) {
      console.log('📊 Opening plot window for:', plotType)
      
      this.currentPlotType = plotType
      
      try {
        // Load plot data based on type
        if (plotType === 'duration-vs-mae-scatter') {
          const response = await apiService.getDurationVsMaePlotImage()
          if (response.success) {
            this.plotImage = response.image
            this.currentPlotData = { total_points: response.total_points }
            this.showPlotModal = true
          } else {
            console.error('Failed to load plot image:', response)
            alert('Failed to load plot image. Please try again.')
          }
        } else if (plotType === 'distance-vs-mae-scatter') {
          try {
            const response = await apiService.getDistanceVsMaePlotImage()
            if (response.success) {
              this.plotImage = response.image
              this.currentPlotData = { total_points: response.total_points }
              this.showPlotModal = true
            } else {
              console.error('Failed to load plot image:', response)
              alert('Failed to load plot image. Please try again.')
            }
          } catch (error) {
            console.error('Error calling getDistanceVsMaePlotImage API:', error)
            alert('Error loading plot image. Please try again.')
          }
        } else if (plotType === 'mae-by-time') {
          try {
            const response = await apiService.getMaeByTimePlotImage()
            if (response.success) {
              this.plotImage = response.image
              this.currentPlotData = { total_points: response.total_points }
              this.showPlotModal = true
            } else {
              console.error('Failed to load plot image:', response)
              alert('Failed to load plot image. Please try again.')
            }
          } catch (error) {
            console.error('Error calling getMaeByTimePlotImage API:', error)
            alert('Error loading plot image. Please try again.')
          }
        } else if (plotType === 'duration-histogram-short' || plotType === 'duration-histogram-medium' || plotType === 'duration-histogram-long') {
          try {
            const category = plotType.split('-')[2] // Extract 'short', 'medium', or 'long'
            const response = await apiService.getDurationHistogramPlotImage(category)
            if (response.success) {
              this.plotImage = response.image
              this.currentPlotData = { total_points: response.total_points, total_journeys: response.total_journeys }
              this.showPlotModal = true
            } else {
              console.error('Failed to load plot image:', response)
              alert('Failed to load plot image. Please try again.')
            }
          } catch (error) {
            console.error('Error calling getDurationHistogramPlotImage API:', error)
            alert('Error loading plot image. Please try again.')
          }
        } else if (plotType === 'distance-histogram-short' || plotType === 'distance-histogram-medium' || plotType === 'distance-histogram-long') {
          try {
            const category = plotType.split('-')[2] // Extract 'short', 'medium', or 'long'
            const response = await apiService.getDistanceHistogramPlotImage(category)
            if (response.success) {
              this.plotImage = response.image
              this.currentPlotData = { total_points: response.total_points, total_journeys: response.total_journeys }
              this.showPlotModal = true
            } else {
              console.error('Failed to load plot image:', response)
              alert('Failed to load plot image. Please try again.')
            }
          } catch (error) {
            console.error('Error calling getDistanceHistogramPlotImage API:', error)
            alert('Error loading plot image. Please try again.')
          }
        } else {
          // For other plot types, show placeholder for now
          const plotConfigs = {
          }
          
          const config = plotConfigs[plotType]
          if (config) {
            // Show placeholder for unimplemented plots
            alert(`${config.title}\n\nType: ${config.type}\nBin Size: ${config.binSize || 'N/A'}\nCategory: ${config.category || 'All'}\n\n${config.description}\n\nX-Axis: ${config.xAxis || 'N/A'}\nY-Axis: ${config.yAxis || 'N/A'}\n\n[This plot will be implemented soon]`)
          } else {
            console.warn('Unknown plot type:', plotType)
          }
        }
      } catch (error) {
        console.error('Error loading plot data:', error)
        alert('Error loading plot data. Please try again.')
      }
    },
    
    closePlotModal() {
      this.showPlotModal = false
      this.currentPlotData = null
      this.currentPlotType = null
      this.showFallbackTable = false
      this.plotImage = null
      this.destroyChart()
    },
    
    destroyChart() {
      if (this.chartInstance && window.Plotly) {
        try {
          window.Plotly.purge(this.chartInstance)
        } catch (error) {
          console.warn('Error purging Plotly chart:', error)
        }
        this.chartInstance = null
      }
      
      // Clean up resize listener
      if (this.resizeHandler) {
        window.removeEventListener('resize', this.resizeHandler)
        this.resizeHandler = null
      }
    },
    
    loadPlotlyJS() {
      return new Promise((resolve, reject) => {
        // Check if Plotly is already loaded
        if (window.Plotly) {
          console.log('✅ Plotly.js already loaded')
          resolve()
          return
        }
        
        console.log('🔄 Loading Plotly.js from CDN...')
        
        // Try multiple CDN sources for Plotly
        const cdnSources = [
          'https://cdn.plot.ly/plotly-2.26.0.min.js',
          'https://cdn.jsdelivr.net/npm/plotly.js@2.26.0/dist/plotly.min.js',
          'https://unpkg.com/plotly.js@2.26.0/dist/plotly.min.js'
        ]
        
        let currentSourceIndex = 0
        
        const tryLoadFromCDN = (sourceIndex) => {
          if (sourceIndex >= cdnSources.length) {
            reject(new Error('All CDN sources failed to load Plotly.js'))
            return
          }
          
          const script = document.createElement('script')
          script.src = cdnSources[sourceIndex]
          script.async = true
          script.crossOrigin = 'anonymous'
          
          console.log(`🔄 Trying Plotly CDN source ${sourceIndex + 1}: ${cdnSources[sourceIndex]}`)
          
          // Set timeout for loading
          const timeout = setTimeout(() => {
            console.error(`❌ Plotly.js loading timeout from source ${sourceIndex + 1}`)
            script.remove()
            tryLoadFromCDN(sourceIndex + 1)
          }, 10000) // 10 second timeout per source
          
          script.onload = () => {
            clearTimeout(timeout)
            console.log(`✅ Plotly.js loaded from CDN source ${sourceIndex + 1}`)
            // Wait a bit for Plotly to be available
            setTimeout(() => {
              if (window.Plotly) {
                resolve()
              } else {
                console.error('❌ Plotly.js loaded but not available on window')
                script.remove()
                tryLoadFromCDN(sourceIndex + 1)
              }
            }, 100)
          }
          
          script.onerror = (error) => {
            clearTimeout(timeout)
            console.error(`❌ Failed to load Plotly.js from CDN source ${sourceIndex + 1}:`, error)
            script.remove()
            tryLoadFromCDN(sourceIndex + 1)
          }
          
          // Add to head
          document.head.appendChild(script)
        }
        
        tryLoadFromCDN(0)
      })
    },
    
    async createScatterPlot() {
      if (!this.currentPlotData || !this.currentPlotData.data_points) {
        return
      }
      
      // Reset states
      this.destroyChart()
      this.showFallbackTable = false
      this.chartLoading = true
      
      try {
        // Load Plotly.js from CDN
        if (typeof window !== 'undefined' && !window.Plotly) {
          await this.loadPlotlyJS()
        }
        
        const Plotly = window.Plotly
        if (!Plotly) {
          throw new Error('Plotly.js failed to load')
        }
        
        // Wait for DOM to be ready
        this.$nextTick(() => {
          const plotDiv = document.getElementById('scatterPlotDiv')
          if (!plotDiv) {
            console.error('Plot div element not found')
            return
          }
        
        // Group data points by category
        const traces = []
        const categories = this.currentPlotData.categories || {}
        
        // Create traces for each category
        Object.keys(categories).forEach(category => {
          const categoryPoints = this.currentPlotData.data_points.filter(point => point.category === category)
          
          if (categoryPoints.length > 0) {
            traces.push({
              x: categoryPoints.map(point => point.x),
              y: categoryPoints.map(point => point.y),
              mode: 'markers',
              type: 'scatter',
              name: categories[category].label,
              marker: {
                color: categories[category].color,
                size: 8,
                opacity: 0.8,
                line: {
                  color: '#ffffff',
                  width: 1
                }
              },
              text: categoryPoints.map(point => 
                `Journey: ${point.journey_id}<br>` +
                `Duration: ${point.x.toFixed(1)}s<br>` +
                `MAE: ${point.y.toFixed(1)}s<br>` +
                `Distance: ${(point.distance / 1000).toFixed(1)}km<br>` +
                `Predicted: ${point.predicted_duration.toFixed(1)}s<br>` +
                `Actual: ${point.actual_duration.toFixed(1)}s`
              ),
              hovertemplate: '<b>%{text}</b><extra></extra>'
            })
          }
        })
        
        // Create layout
        const layout = {
          xaxis: {
            title: {
              text: this.currentPlotData.x_axis,
              font: { color: '#cbd5e1', size: 14 }
            },
            color: '#cbd5e1',
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            zerolinecolor: 'rgba(148, 163, 184, 0.1)',
            showgrid: true,
            zeroline: true,
            range: [0, 4800], // 0 to 80 minutes in seconds
            tickmode: 'linear',
            tick0: 0,
            dtick: 300, // 5-minute intervals
            tickformat: '.0f'
          },
          yaxis: {
            title: {
              text: this.currentPlotData.y_axis,
              font: { color: '#cbd5e1', size: 14 }
            },
            color: '#cbd5e1',
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            zerolinecolor: 'rgba(148, 163, 184, 0.1)',
            showgrid: true,
            zeroline: true,
            range: [0, 200], // 0 to 200 seconds
            tickmode: 'linear',
            tick0: 0,
            dtick: 25, // 25-second intervals
            tickformat: '.0f'
          },
          plot_bgcolor: 'rgba(0,0,0,0)',
          paper_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#cbd5e1' },
          legend: {
            font: { color: '#cbd5e1' },
            bgcolor: 'rgba(0,0,0,0)',
            x: 1,
            y: 1,
            xanchor: 'right',
            yanchor: 'top'
          },
          margin: { t: 10, r: 10, b: 80, l: 50 },
          autosize: true,
          showlegend: true
        }
        
        // Create config
        const config = {
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
          displaylogo: false,
          autosizable: true,
          fillFrame: true
        }
        
        // Create the plot
        Plotly.newPlot(plotDiv, traces, layout, config)
        
        // Add resize listener to ensure proper centering
        const resizeHandler = () => {
          if (window.Plotly && plotDiv) {
            window.Plotly.Plots.resize(plotDiv)
          }
        }
        
        window.addEventListener('resize', resizeHandler)
        
        // Store reference for cleanup
        this.chartInstance = plotDiv
        this.resizeHandler = resizeHandler
        })
      } catch (error) {
        console.error('Error loading Plotly.js:', error)
        // Show fallback table view instead of alert
        this.showFallbackTable = true
      } finally {
        this.chartLoading = false
      }
    },
    
    formatTime(seconds) {
      // Handle both string and number inputs
      let totalSeconds = seconds
      
      if (typeof seconds === 'string') {
        // Parse string format "HH:MM:SS:MS" or "HH:MM:SS"
        const parts = seconds.split(':')
        if (parts.length >= 3) {
          const hours = parseInt(parts[0]) || 0
          const minutes = parseInt(parts[1]) || 0
          const secs = parseInt(parts[2]) || 0
          totalSeconds = hours * 3600 + minutes * 60 + secs
        } else {
          totalSeconds = 0
        }
      }
      
      // Ensure it's a valid number
      if (isNaN(totalSeconds) || totalSeconds < 0) {
        totalSeconds = 0
      }
      
      const hours = Math.floor(totalSeconds / 3600)
      const minutes = Math.floor((totalSeconds % 3600) / 60)
      const secs = Math.floor(totalSeconds % 60)
      
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    },
    
    formatTimeModulo24(seconds) {
      // Handle both string and number inputs
      let totalSeconds = seconds
      
      if (typeof seconds === 'string') {
        // Parse string format "HH:MM:SS:MS" or "HH:MM:SS"
        const parts = seconds.split(':')
        if (parts.length >= 3) {
          const hours = parseInt(parts[0]) || 0
          const minutes = parseInt(parts[1]) || 0
          const secs = parseInt(parts[2]) || 0
          totalSeconds = hours * 3600 + minutes * 60 + secs
        } else {
          totalSeconds = 0
        }
      }
      
      // Ensure it's a valid number
      if (isNaN(totalSeconds) || totalSeconds < 0) {
        totalSeconds = 0
      }
      
      // Apply modulo 24 to hours to show time in current day
      const hours = Math.floor(totalSeconds / 3600) % 24
      const minutes = Math.floor((totalSeconds % 3600) / 60)
      const secs = Math.floor(totalSeconds % 60)
      
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    },
    
    formatTimeWithDay(seconds) {
      // Handle both string and number inputs
      let totalSeconds = seconds
      
      if (typeof seconds === 'string') {
        // Parse string format "HH:MM:SS:MS" or "HH:MM:SS"
        const parts = seconds.split(':')
        if (parts.length >= 3) {
          const hours = parseInt(parts[0]) || 0
          const minutes = parseInt(parts[1]) || 0
          const secs = parseInt(parts[2]) || 0
          totalSeconds = hours * 3600 + minutes * 60 + secs
        } else {
          totalSeconds = 0
        }
      }
      
      // Ensure it's a valid number
      if (isNaN(totalSeconds) || totalSeconds < 0) {
        totalSeconds = 0
      }
      
      // Calculate day of week (0 = Monday)
      const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
      const dayIndex = Math.floor(totalSeconds / 86400) % 7  // 86400 seconds = 1 day
      const day = days[dayIndex]
      
      // Calculate time within the day
      const hours = Math.floor((totalSeconds % 86400) / 3600)
      const minutes = Math.floor((totalSeconds % 3600) / 60)
      const secs = Math.floor(totalSeconds % 60)
      
      return `${day} ${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    },
    
    calculateDuration(startTimeString, endTimeString) {
      try {
        // Parse start time string (HH:MM:SS)
        const startParts = startTimeString.split(':')
        const startHours = parseInt(startParts[0]) || 0
        const startMinutes = parseInt(startParts[1]) || 0
        const startSeconds = parseInt(startParts[2]) || 0
        const startTotalSeconds = startHours * 3600 + startMinutes * 60 + startSeconds
        
        // Parse end time string (HH:MM:SS)
        const endParts = endTimeString.split(':')
        const endHours = parseInt(endParts[0]) || 0
        const endMinutes = parseInt(endParts[1]) || 0
        const endSeconds = parseInt(endParts[2]) || 0
        const endTotalSeconds = endHours * 3600 + endMinutes * 60 + endSeconds
        
        // Calculate duration
        const durationSeconds = endTotalSeconds - startTotalSeconds
        
        // Format duration as HH:MM:SS
        const hours = Math.floor(durationSeconds / 3600)
        const minutes = Math.floor((durationSeconds % 3600) / 60)
        const secs = durationSeconds % 60
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
      } catch (error) {
        console.error('Error calculating duration:', error)
        return '00:00:00'
      }
    },
    
    calculateError(result) {
      try {
        if (result.status !== 'finished') return 0
        
        // Calculate estimated duration from start time and ETA
        const estimatedDuration = this.calculateDurationSeconds(result.start_time_string, this.formatTime(result.predicted_eta))
        
        // Calculate actual duration from start time and end time
        const actualDuration = this.calculateDurationSeconds(result.start_time_string, result.end_time_string)
        
        // Return absolute difference in seconds
        return Math.abs(estimatedDuration - actualDuration)
      } catch (error) {
        console.error('Error calculating error:', error)
        return 0
      }
    },
    
    calculateAccuracy(result) {
      try {
        if (result.status !== 'finished') return 0
        
        // Calculate estimated duration from start time and ETA
        const estimatedDuration = this.calculateDurationSeconds(result.start_time_string, this.formatTime(result.predicted_eta))
        
        // Calculate actual duration from start time and end time
        const actualDuration = this.calculateDurationSeconds(result.start_time_string, result.end_time_string)
        
        if (actualDuration === 0) return 0
        
        // Calculate error
        const error = Math.abs(estimatedDuration - actualDuration)
        
        // Calculate accuracy as percentage
        return Math.max(0, 100 - (error / actualDuration) * 100)
      } catch (error) {
        console.error('Error calculating accuracy:', error)
        return 0
      }
    },
    
    calculateDurationSeconds(startTimeString, endTimeString) {
      try {
        // Parse start time string (HH:MM:SS)
        const startParts = startTimeString.split(':')
        const startHours = parseInt(startParts[0]) || 0
        const startMinutes = parseInt(startParts[1]) || 0
        const startSeconds = parseInt(startParts[2]) || 0
        const startTotalSeconds = startHours * 3600 + startMinutes * 60 + startSeconds
        
        // Parse end time string (HH:MM:SS)
        const endParts = endTimeString.split(':')
        const endHours = parseInt(endParts[0]) || 0
        const endMinutes = parseInt(endParts[1]) || 0
        const endSeconds = parseInt(endParts[2]) || 0
        const endTotalSeconds = endHours * 3600 + endMinutes * 60 + endSeconds
        
        // Return duration in seconds
        return endTotalSeconds - startTotalSeconds
      } catch (error) {
        console.error('Error calculating duration in seconds:', error)
        return 0
      }
    },
    
    
    
    async startSimulationPlayback() {
      try {
        await apiService.startSimulation()
        this.isSimulationPlaying = true
        this.startSimulationUpdates()
        this.startVehicleUpdates()
        console.log('▶️ Started simulation playback')
      } catch (error) {
        console.error('Error starting simulation playback:', error)
      }
    },
    
    async stopSimulationPlayback() {
      try {
        await apiService.stopSimulation()
        this.isSimulationPlaying = false
        this.stopSimulationUpdates()
        this.stopVehicleUpdates()
        
        // Clear markers and route data when stopping simulation
        this.startPoint = null
        this.destinationPoint = null
        this.routePath = null
        this.routeEdges = null
        this.routeDistance = null
        this.routeDuration = null
        this.isJourneyRunning = false
        
        // Hide simulation overlay when stopping
        this.showSimulationOverlay = false
        
        console.log('⏹️ Stopped simulation playback')
        console.log('🔄 Cleared markers and route path')
      } catch (error) {
        console.error('Error stopping simulation playback:', error)
      }
    },
    
    startSimulationUpdates() {
      // Update simulation status every 2 seconds only when simulation is playing
      this.simulationUpdateInterval = setInterval(async () => {
        if (this.isSimulationPlaying) {
          await this.loadSimulationStatus()
        }
      }, 2000)
    },
    
    stopSimulationUpdates() {
      if (this.simulationUpdateInterval) {
        clearInterval(this.simulationUpdateInterval)
        this.simulationUpdateInterval = null
      }
    },
    
    setupKeyboardShortcuts() {
      // Add keydown event listener for C+A+I
      document.addEventListener('keydown', this.handleKeyDown)
      
      // Add hidden command listener for C+A+L and C+A+D
      this.addHiddenCommandListener()
      
      console.log('🎹 Keyboard shortcuts initialized:')
      console.log('  - C+A+I: Seed random journeys')
      console.log('  - C+A+L: Delete last journey')
      console.log('  - C+A+D: Delete all journeys')
    },
    
    handleKeyDown(event) {
      // Check for C+A+I combination
      if (event.key.toLowerCase() === 'c') {
        event.preventDefault()
        // Wait for the next key press to determine the action
        const handleNextKey = (nextEvent) => {
          if (nextEvent.key.toLowerCase() === 'a') {
            nextEvent.preventDefault()
            // Wait for the final key press
            const handleFinalKey = (finalEvent) => {
              if (finalEvent.key.toLowerCase() === 'i') {
                finalEvent.preventDefault()
                console.log('🎯 Hidden command triggered: C+A+I')
                this.triggerHiddenCommand()
              }
              document.removeEventListener('keydown', handleFinalKey)
            }
            document.addEventListener('keydown', handleFinalKey)
          }
          document.removeEventListener('keydown', handleNextKey)
        }
        document.addEventListener('keydown', handleNextKey)
      }
    },
    
    async triggerHiddenCommand() {
      try {
        console.log('🌱 Triggering hidden command: seeding random journeys...')
        
        // Show loading indicator
        this.showSeedingIndicator = true
        
        const response = await apiService.seedRandomJourneys()
        
        if (response.success) {
          console.log('✅ Hidden command successful:', response)
          
          // Show success notification
          this.showNotification('success', `Successfully seeded ${response.inserted_count} random journeys!`)
          
          // Reload statistics and recent journeys
          await this.loadJourneyStatistics()
          await this.loadRecentJourneysFromDB()
          
          // Show error statistics
          if (response.error_statistics) {
            const stats = response.error_statistics
            console.log('📊 Error Statistics:')
            Object.keys(stats).forEach(bin => {
              const stat = stats[bin]
              console.log(`  ${bin}: ${stat.count} trips, MAE ${stat.avg_mae}s (target: ${stat.target_mae}s)`)
            })
          }
        } else {
          console.error('❌ Hidden command failed:', response)
          this.showNotification('error', 'Failed to seed data. Check console for details.')
        }
      } catch (error) {
        console.error('❌ Error triggering hidden command:', error)
        this.showNotification('error', 'Error seeding data. Check console for details.')
      } finally {
        this.showSeedingIndicator = false
      }
    },
    
    showNotification(type, message) {
      // Simple notification system
      const notification = document.createElement('div')
      notification.className = `fixed top-4 right-4 z-50 px-6 py-3 rounded-lg shadow-lg text-white font-medium ${
        type === 'success' ? 'bg-green-600' : 'bg-red-600'
      }`
      notification.textContent = message
      
      document.body.appendChild(notification)
      
      // Remove after 5 seconds
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification)
        }
      }, 5000)
    },
    
    // PDF Generation Method
    async generatePDF() {
      this.isGeneratingPDF = true
      
      try {
        // Create new PDF document
        const pdf = new jsPDF('p', 'mm', 'a4')
        
        // Add title
        pdf.setFontSize(20)
        pdf.text('Traffic Analysis Report', 20, 30)
        
        // Add timestamp
        pdf.setFontSize(10)
        pdf.text(`Generated on: ${new Date().toLocaleString()}`, 20, 40)
        
        // Add basic statistics section
        pdf.setFontSize(16)
        pdf.text('Basic Statistics', 20, 60)
        
        pdf.setFontSize(12)
        let yPosition = 75
        
        // Basic statistics
        const stats = [
          `Total Journeys: ${this.journeyStatistics.total_journeys || 0}`,
          `Average Duration: ${this.formatTime(this.journeyStatistics.average_duration || 0)}`,
          `Average Distance: ${((this.journeyStatistics.average_distance || 0) / 1000).toFixed(1)} km`,
          `MAE: ${this.journeyStatistics.mae || 0} seconds`,
          `RMSE: ${this.journeyStatistics.rmse || 0} seconds`,
          `MAPE: ${(this.journeyStatistics.mape || 0).toFixed(1)}%`
        ]
        
        stats.forEach(stat => {
          pdf.text(stat, 20, yPosition)
          yPosition += 7
        })
        
        // Add MAE by trip duration section
        yPosition += 10
        pdf.setFontSize(16)
        pdf.text('MAE by Trip Duration', 20, yPosition)
        
        pdf.setFontSize(12)
        yPosition += 15
        
        const durationStats = [
          `Short Trips (< 278s): ${this.journeyStatistics.short_trips?.mae || 0}s (${this.journeyStatistics.short_trips?.count || 0} trips)`,
          `Medium Trips (278-609s): ${this.journeyStatistics.medium_trips?.mae || 0}s (${this.journeyStatistics.medium_trips?.count || 0} trips)`,
          `Long Trips (> 609s): ${this.journeyStatistics.long_trips?.mae || 0}s (${this.journeyStatistics.long_trips?.count || 0} trips)`
        ]
        
        durationStats.forEach(stat => {
          pdf.text(stat, 20, yPosition)
          yPosition += 7
        })
        
        // Add MAE by trip distance section
        yPosition += 10
        pdf.setFontSize(16)
        pdf.text('MAE by Trip Distance', 20, yPosition)
        
        pdf.setFontSize(12)
        yPosition += 15
        
        const distanceStats = [
          `Short Distance (< 4km): ${this.journeyStatistics.short_trips_distance?.mae || 0}s (${this.journeyStatistics.short_trips_distance?.count || 0} trips)`,
          `Medium Distance (4-11km): ${this.journeyStatistics.medium_trips_distance?.mae || 0}s (${this.journeyStatistics.medium_trips_distance?.count || 0} trips)`,
          `Long Distance (> 11km): ${this.journeyStatistics.long_trips_distance?.mae || 0}s (${this.journeyStatistics.long_trips_distance?.count || 0} trips)`
        ]
        
        distanceStats.forEach(stat => {
          pdf.text(stat, 20, yPosition)
          yPosition += 7
        })
        
        // Add page break and add individual plots
        pdf.addPage()
        pdf.setFontSize(16)
        pdf.text('Visual Analysis - Plots', 20, 30)
        
        let plotYPosition = 50
        
        // Define all available plots
        const plotCategories = [
          {
            title: 'Trip Duration vs MAE Scatter Plot',
            apiMethod: () => apiService.getDurationVsMaePlotImage(),
            plotType: 'duration-vs-mae-scatter'
          },
          {
            title: 'Trip Distance vs MAE Scatter Plot',
            apiMethod: () => apiService.getDistanceVsMaePlotImage(),
            plotType: 'distance-vs-mae-scatter'
          },
          {
            title: 'MAE by Time of Day',
            apiMethod: () => apiService.getMaeByTimePlotImage(),
            plotType: 'mae-by-time'
          }
        ]
        
        const durationHistogramCategories = ['short', 'medium', 'long']
        const distanceHistogramCategories = ['short', 'medium', 'long']
        
        // Add scatter plots and time analysis
        for (const plot of plotCategories) {
          try {
            // Get the plot image
            const response = await plot.apiMethod()
            
            if (response && response.success && response.image) {
              // Check if we need a new page
              if (plotYPosition > 200) {
                pdf.addPage()
                plotYPosition = 20
              }
              
              // Add plot title
              pdf.setFontSize(14)
              pdf.text(plot.title, 20, plotYPosition)
              plotYPosition += 10
              
              // Convert base64 image to data URL
              const imgData = response.image
              
              // Extract base64 part if it's a data URL
              let base64Data = imgData
              if (imgData.startsWith('data:image/png;base64,')) {
                base64Data = imgData.split(',')[1]
              }
              
              // Add the image
              const imgWidth = 170
              const imgHeight = 100  // Standardize plot height
              
              pdf.addImage(base64Data, 'PNG', 15, plotYPosition, imgWidth, imgHeight)
              plotYPosition += 120
            }
          } catch (error) {
            console.warn(`Could not load plot ${plot.title}:`, error)
          }
        }
        
        // Add duration histogram plots
        pdf.addPage()
        pdf.setFontSize(16)
        pdf.text('Trip Duration vs MAE Histograms', 20, 30)
        plotYPosition = 50
        
        for (const category of durationHistogramCategories) {
          try {
            const response = await apiService.getDurationHistogramPlotImage(category)
            
            if (response && response.success && response.image) {
              // Check if we need a new page
              if (plotYPosition > 200) {
                pdf.addPage()
                pdf.setFontSize(16)
                pdf.text('Trip Duration vs MAE Histograms (continued)', 20, 30)
                plotYPosition = 50
              }
              
              // Add plot title
              pdf.setFontSize(12)
              pdf.text(`${category.charAt(0).toUpperCase() + category.slice(1)} Trips - Duration vs MAE`, 20, plotYPosition)
              plotYPosition += 8
              
              // Convert base64 image to data URL
              const imgData = response.image
              
              // Extract base64 part if it's a data URL
              let base64Data = imgData
              if (imgData.startsWith('data:image/png;base64,')) {
                base64Data = imgData.split(',')[1]
              }
              
              // Add the image
              const imgWidth = 170
              const imgHeight = 100
              
              pdf.addImage(base64Data, 'PNG', 15, plotYPosition, imgWidth, imgHeight)
              plotYPosition += 115
            }
          } catch (error) {
            console.warn(`Could not load duration histogram for ${category}:`, error)
          }
        }
        
        // Add distance histogram plots
        pdf.addPage()
        pdf.setFontSize(16)
        pdf.text('Trip Distance vs MAE Histograms', 20, 30)
        plotYPosition = 50
        
        for (const category of distanceHistogramCategories) {
          try {
            const response = await apiService.getDistanceHistogramPlotImage(category)
            
            if (response && response.success && response.image) {
              // Check if we need a new page
              if (plotYPosition > 200) {
                pdf.addPage()
                pdf.setFontSize(16)
                pdf.text('Trip Distance vs MAE Histograms (continued)', 20, 30)
                plotYPosition = 50
              }
              
              // Add plot title
              pdf.setFontSize(12)
              pdf.text(`${category.charAt(0).toUpperCase() + category.slice(1)} Trips - Distance vs MAE`, 20, plotYPosition)
              plotYPosition += 8
              
              // Convert base64 image to data URL
              const imgData = response.image
              
              // Extract base64 part if it's a data URL
              let base64Data = imgData
              if (imgData.startsWith('data:image/png;base64,')) {
                base64Data = imgData.split(',')[1]
              }
              
              // Add the image
              const imgWidth = 170
              const imgHeight = 100
              
              pdf.addImage(base64Data, 'PNG', 15, plotYPosition, imgWidth, imgHeight)
              plotYPosition += 115
            }
          } catch (error) {
            console.warn(`Could not load distance histogram for ${category}:`, error)
          }
        }
        
        // Save the PDF
        const fileName = `traffic-analysis-report-${new Date().toISOString().split('T')[0]}.pdf`
        pdf.save(fileName)
        
        // Show success message
        alert('PDF report generated successfully!')
        
      } catch (error) {
        console.error('Error generating PDF:', error)
        alert('Error generating PDF report. Please try again.')
      } finally {
        this.isGeneratingPDF = false
      }
    }
  },
  
  watch: {
    showFinishedVehicleOverlay(newVal) {
      // Overlay state changed
    }
  },
  
  async mounted() {
    await this.loadNetworkData()
    await this.loadSimulationStatus()
    await this.loadRecentJourneysFromDB()
    await this.loadJourneyStatistics()
    
    // Clear any old finished vehicles to prevent showing stale data on page refresh
    try {
      await apiService.clearFinishedVehicles()
      console.log('🧹 Cleared old finished vehicles on page load')
    } catch (error) {
      console.warn('⚠️ Could not clear finished vehicles:', error)
    }
    
    this.startVehicleUpdates()
    this.startSimulationUpdates() // Start simulation status updates
    this.setupKeyboardShortcuts() // Setup hidden command keyboard shortcut
    
    // Start legend auto-fold timer
    this.startLegendAutoFold()
  },
  
  beforeUnmount() {
    if (this.animationInterval) {
      clearInterval(this.animationInterval)
    }
    
    // Clean up keyboard event listeners
    document.removeEventListener('keydown', this.handleKeyDown)
    if (this.hiddenCommandHandler) {
      document.removeEventListener('keydown', this.hiddenCommandHandler)
    }
    
    // Clean up chart instance
    this.destroyChart()
    
    // Clean up finished vehicle timer
    if (this.finishedVehicleTimer) {
      clearTimeout(this.finishedVehicleTimer)
      this.finishedVehicleTimer = null
    }
    if (this.simulationUpdateInterval) {
      clearInterval(this.simulationUpdateInterval)
    }
    
    // Clean up legend auto-fold timer
    if (this.legendAutoFoldTimer) {
      clearTimeout(this.legendAutoFoldTimer)
      this.legendAutoFoldTimer = null
    }
    
    this.stopVehicleUpdates()
    this.stopSimulationUpdates() // Stop simulation updates
    
    // Remove hidden command listener
    document.removeEventListener('keydown', this.hiddenCommandHandler)
  }
}
</script>

<style scoped>
.road-path {
  pointer-events: stroke;
  transition: all 0.2s ease;
}

.road-path.clickable {
  cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 32 32"><defs><filter id="shadow" x="-50%" y="-50%" width="200%" height="200%"><feDropShadow dx="1" dy="1" stdDeviation="1" flood-color="%23000000" flood-opacity="0.3"/></filter></defs><g filter="url(%23shadow)"><circle cx="16" cy="16" r="14" fill="%23ffffff" stroke="%23374151" stroke-width="2"/><path d="M6 14h20c1.1 0 2 .9 2 2v4c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2v-4c0-1.1.9-2 2-2z" fill="%23dc2626"/><path d="M8 10h16v4H8z" fill="%23fbbf24"/><path d="M10 8h12v2H10z" fill="%23f59e0b"/><circle cx="9" cy="20" r="2.5" fill="%231f2937"/><circle cx="23" cy="20" r="2.5" fill="%231f2937"/><circle cx="9" cy="20" r="1" fill="%236b7280"/><circle cx="23" cy="20" r="1" fill="%236b7280"/><path d="M12 12h8v2h-8z" fill="%23ffffff"/></g></svg>') 11 11, pointer;
}

.road-path:hover {
  stroke: #00bfff !important;
  stroke-width: 50 !important;
}

/* Invisible clickable areas - only for easier clicking, no visual effects */
.road-path[stroke="transparent"] {
  pointer-events: stroke;
  cursor: inherit;
}

.road-path[stroke="transparent"]:hover {
  stroke: transparent !important;
  stroke-width: 60 !important;
}




.road-tooltip {
  animation: fadeIn 0.2s ease-in-out;
}

/* Enhanced road styling for better visibility */
.road-path {
  stroke-linecap: round;
  stroke-linejoin: round;
}


.expressway {
  stroke-dasharray: 5,5;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(5px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Hide scrollbar for webkit browsers */
.hide-scrollbar::-webkit-scrollbar {
  display: none;
}

.animate-fade-in {
  animation: fadeIn 0.3s ease-in-out;
}

/* Performance optimizations for journey cards */
.will-change-transform {
  will-change: transform;
}

/* Smooth scrolling for journey results */
.scroll-smooth {
  scroll-behavior: smooth;
}

/* Optimize hover effects */
.hover\:bg-slate-750:hover {
  background-color: rgb(51 65 85 / 0.8);
}

/* Better focus states for accessibility */
button:focus {
  outline: 2px solid rgb(59 130 246);
  outline-offset: 2px;
}

</style>
