<template>
  <div class="app-container">
    <!-- Navigation -->
    <nav class="bg-blue-900 text-white shadow-lg">
      <div class="w-full px-2 xs:px-3 sm:px-4 md:px-6 lg:px-8 xl:px-12 2xl:px-16 3xl:px-20 4xl:px-24">
        <div class="header-grid">
          <!-- Left side - Lab name -->
          <div class="header-left">
            <router-link to="/" class="header-logo">
              SmartTransportation Lab
            </router-link>
          </div>
          
          <!-- Right side - ETA text -->
          <div class="header-right">
            <span class="header-subtitle">
              <span class="header-mobile-text">Real time ETA prediction</span>
              <span class="header-desktop-text">Real time ETA prediction model performance testing</span>
            </span>
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content Section -->
    <div class="main-content">
      <!-- Landscape Orientation Message (Desktop/Landscape) -->
      <div class="landscape-message">
        <div class="landscape-content">
          <div class="phone-icon">📱</div>
          <h2>Please Rotate Your Device</h2>
          <p>For the best simulation experience, please rotate your device to portrait orientation.</p>
          <div class="rotation-hint">
            <div class="arrow">↻</div>
            <span>Turn your device upright</span>
          </div>
        </div>
      </div>

      <!-- Simulation Layout (Portrait Only) -->
      <div class="simulation-layout">
        <div class="content-grid">
          <!-- Analysis Section - 20% -->
          <div class="analysis-section">
            <!-- Model Performance Analysis -->
            <div class="bg-slate-900 bg-opacity-98 rounded-lg shadow-xl h-full overflow-hidden">
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
          </div>
          
          <!-- Map Section - 60% -->
          <div class="map-section">
            <!-- SUMO Network Container -->
            <div 
              class="map-container"
              ref="mapContainer"
            >
              <!-- Simulation Time Display - Floating above network -->
              <div class="simulation-time-display">
                {{ formatTimeWithDay(simulationTime) }}
              </div>
              <!-- SUMO Network Visualization -->
              <svg 
                ref="networkSvg"
                :viewBox="svgViewBox" 
                width="100%" 
                height="100%" 
                class="absolute inset-0 cursor-crosshair map-container" 
                style="background: #f1f5f9;"
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
                        @touchstart="handleEdgeClick(edge, $event)"
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
                        @touchstart="handleEdgeClick(edge, $event)"
                      />
                      <!-- Regular roads -->
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
                        @touchstart="handleEdgeClick(edge, $event)"
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
                        @touchstart="handleEdgeClick(edge, $event)"
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
                        @touchstart="handleEdgeClick(edge, $event)"
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
                        @touchstart="handleEdgeClick(edge, $event)"
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
                  <!-- User-defined vehicles (sports car image) -->
                  <g 
                    v-for="vehicle in activeVehicles.filter(v => !isNaN(v.x) && !isNaN(v.y) && v.type === 'user_defined')"
                    :key="vehicle.id"
                    :transform="getVehicleTransform(vehicle)"
                    class="vehicle-marker"
                    opacity="0.8"
                  >
                    <image 
                      :x="-216" 
                      :y="-108" 
                      width="432" 
                      height="216" 
                      href="/images/car.png"
                      preserveAspectRatio="xMidYMid meet"
                    />
                  </g>
              </g>
            </svg>

            <!-- Desktop/Tablet Landscape Layout - Positioned 25% towards left -->
            <div class="hidden sm:block absolute top-4 left-1/4 transform -translate-x-2/3 pointer-events-none z-10 landscape-only">
              <div class="bg-white bg-opacity-70 rounded-xl shadow-xl p-4 text-center max-w-xs animate-fade-in">
                <div class="text-2xl mb-2 animate-bounce">
                  <span v-if="!startPoint">📍</span>
                  <span v-else-if="!destinationPoint">🎯</span>
                  <span v-else-if="!isJourneyRunning">🚀</span>
                  <span v-else>🚗</span>
                </div>
                <h3 class="text-lg font-bold text-gray-800 mb-1">
                  <span v-if="!startPoint">Set Starting Point</span>
                  <span v-else-if="!destinationPoint">Set Destination</span>
                  <span v-else-if="!isJourneyRunning">Ready to Start!</span>
                  <span v-else>Journey in Progress</span>
                </h3>
                <p class="text-sm text-gray-600">
                  {{ getInstructionText() }}
                </p>
              </div>
            </div>

            <!-- Map Legend -->
            <div v-if="showLegend" class="absolute top-2 left-2 sm:top-4 sm:left-4 bg-white bg-opacity-95 rounded-lg shadow-lg pointer-events-auto animate-fade-in hidden lg:block z-20" style="max-width: 180px;">
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

            <!-- Control Buttons - Top Right Overlay -->
            <div class="absolute top-2 right-2 sm:top-4 sm:right-4 z-20 pointer-events-auto">
              <div class="flex flex-col items-stretch gap-1 sm:gap-2">
                <!-- Reset Points Button - Show when points are selected but journey not running -->
                <button 
                  v-if="startPoint && !isJourneyRunning"
                  @click="resetPoints"
                  class="bg-gray-600 hover:bg-gray-700 text-white px-2 py-1 sm:px-4 sm:py-2 rounded-md sm:rounded-lg font-semibold transition duration-300 text-xs sm:text-sm shadow-lg"
                >
                  🔄 Reset Points
                </button>
                
                <!-- Start Journey Button - Show when both points are set but journey not running -->
                <button 
                  v-if="startPoint && destinationPoint && !isJourneyRunning"
                  @click="handleMainButtonClick()"
                  :class="getMainButtonClass()"
                  class="text-white px-2 py-1 sm:px-4 sm:py-2 rounded-md sm:rounded-lg font-semibold transition duration-300 text-xs sm:text-sm shadow-lg"
                >
                  {{ getMainButtonText() }}
                </button>
              </div>
            </div>
          </div>
        </div>
          
          <!-- Results Section - 20% -->
          <div class="results-section">
            <!-- Results Summary Section -->
            <div class="bg-slate-900 bg-opacity-98 rounded-lg shadow-xl h-full overflow-hidden">
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
                
                <div v-else class="space-y-1 sm:space-y-1.5">
                  <div v-for="(result, index) in vehicleResults" :key="result.vehicle_id" 
                       class="border border-slate-600 rounded-lg p-1.5 sm:p-2 bg-slate-800 hover:bg-slate-750 transition-all duration-200 hover:shadow-lg hover:border-slate-500 will-change-transform"
                       :class="{ 'opacity-50': result.status === 'running' && !isSimulationPlaying }">
                    <!-- Status Header -->
                    <div class="flex items-center justify-between mb-1 sm:mb-2">
                      <div class="flex items-center space-x-1 sm:space-x-2">
                        <span class="text-xs sm:text-xs font-medium text-slate-300">Journey #{{ totalJourneyCount - index }}</span>
        </div>
                      <span class="text-xs px-2 sm:px-3 py-0.5 sm:py-1 rounded-full font-medium flex items-center space-x-1"
                            :class="result.status === 'finished' ? 'bg-emerald-600 text-emerald-100' : 'bg-blue-600 text-blue-100'">
                        <span v-if="result.status === 'finished'">✓</span>
                        <span class="font-mono text-xs">{{ result.status === 'finished' ? 'Completed' : 'Running' }}</span>
                        <span v-if="result.status === 'running'" class="ml-1 font-mono text-xs">
                          {{ formatTime(Math.max(0, simulationTime - result.start_time)) }}
                        </span>
                      </span>
                    </div>
                    
                    <!-- Top Section: Start and Distance -->
                    <div class="grid grid-cols-2 gap-1 sm:gap-1.5 mb-1 sm:mb-2">
                      <!-- Start Time -->
                      <div class="bg-slate-700/50 rounded p-1 sm:p-1.5">
                        <div class="text-xs text-slate-400 mb-0.5 font-medium flex items-center">
                          <span class="mr-0.5 sm:mr-1">🕐</span>
                          <span class="hidden sm:inline">Start</span>
                        </div>
                        <div class="text-xs font-mono text-slate-100">
                          {{ result.start_time_string ? formatTimeWithDay(result.start_time_string) : formatTimeWithDay(result.start_time) }}
                        </div>
                      </div>
                      
                      <!-- Distance -->
                      <div class="bg-slate-700/50 rounded p-1 sm:p-1.5">
                        <div class="text-xs text-slate-400 mb-0.5 font-medium flex items-center">
                          <span class="mr-0.5 sm:mr-1">📏</span>
                          <span class="hidden sm:inline">Distance</span>
                        </div>
                        <div class="text-xs font-mono text-slate-100">
                          {{ (result.distance / 1000).toFixed(2) }}km
                        </div>
                      </div>
                    </div>
                    
                    <!-- Line Separator -->
                    <div class="border-t border-slate-600 mb-1 sm:mb-2"></div>
                    
                    <!-- Middle Section: Prediction -->
                    <div class="text-xs font-medium text-slate-300 mb-1 sm:mb-1.5 flex items-center">
                      <span class="mr-0.5 sm:mr-1">🎯</span>
                      <span class="hidden sm:inline">Prediction</span>
                    </div>
                    <div class="grid grid-cols-2 gap-1 sm:gap-1.5 mb-1 sm:mb-2">
                      <!-- ETA -->
                      <div class="bg-slate-700/50 rounded p-1 sm:p-1.5">
                        <div class="text-xs text-slate-400 mb-0.5 font-medium flex items-center">
                          <span class="mr-0.5 sm:mr-1">🎯</span>
                          <span class="hidden sm:inline">ETA</span>
                        </div>
                        <div class="text-xs font-mono text-slate-100">
                          {{ formatTimeModulo24(result.predicted_eta) }}
                        </div>
                      </div>
                      
                      <!-- Estimated Duration -->
                      <div class="bg-slate-700/50 rounded p-1 sm:p-1.5">
                        <div class="text-xs text-slate-400 mb-0.5 font-medium flex items-center">
                          <span class="mr-0.5 sm:mr-1">⏱️</span>
                          <span class="hidden sm:inline">Est. Duration</span>
                        </div>
                        <div class="text-xs font-mono text-slate-100">
                          {{ calculateDuration(result.start_time_string, formatTime(result.predicted_eta)) }}
                        </div>
                      </div>
                    </div>
                    
                    <!-- Line Separator -->
                    <div class="border-t border-slate-600 mb-1 sm:mb-2"></div>
                    
                    <!-- Results (only if finished) -->
                    <div v-if="result.status === 'finished'" class="space-y-1 sm:space-y-1.5">
                      <div class="text-xs font-medium text-slate-300 mb-1 sm:mb-1.5 flex items-center">
                        <span class="mr-0.5 sm:mr-1">📊</span>
                        <span class="hidden sm:inline">Results</span>
                      </div>
                      
                      <!-- First Line: Arrival Time and Duration -->
                      <div class="grid grid-cols-2 gap-1 sm:gap-1.5 mb-1 sm:mb-1.5">
                        <!-- Arrival Time -->
                        <div class="bg-slate-700/50 rounded p-1 sm:p-1.5">
                          <div class="text-xs text-slate-400 mb-0.5 font-medium">
                            <span class="hidden sm:inline">Arrival Time</span>
                            <span class="sm:hidden">Arrival</span>
                          </div>
                          <div class="text-xs font-mono text-slate-100">
                            {{ result.end_time_string ? formatTimeModulo24(result.end_time_string) : formatTimeModulo24(result.end_time) }}
                          </div>
                        </div>
                        
                        <!-- Duration -->
                        <div class="bg-slate-700/50 rounded p-1 sm:p-1.5">
                          <div class="text-xs text-slate-400 mb-0.5 font-medium">Duration</div>
                          <div class="text-xs font-mono text-slate-100">
                            {{ formatTime(result.actual_duration) }}
                          </div>
                        </div>
                      </div>
                      
                      <!-- Second Line: Error and Accuracy -->
                      <div class="grid grid-cols-2 gap-1 sm:gap-1.5">
                        <!-- Absolute Error -->
                        <div class="bg-slate-700/50 rounded p-1 sm:p-1.5">
                          <div class="text-xs text-slate-400 mb-0.5 font-medium">Error</div>
                          <div class="text-xs font-mono px-1 sm:px-1.5 py-0.5 rounded" 
                               :class="calculateError(result) < 30 ? 'text-emerald-300 bg-emerald-900/30' : calculateError(result) < 60 ? 'text-yellow-300 bg-yellow-900/30' : 'text-red-300 bg-red-900/30'">
                            {{ Math.round(calculateError(result)) }}s
                          </div>
                        </div>
                        
                        <!-- Accuracy -->
                        <div class="bg-slate-700/50 rounded p-1 sm:p-1.5">
                          <div class="text-xs text-slate-400 mb-0.5 font-medium">Accuracy</div>
                          <div class="text-xs font-mono px-1 sm:px-1.5 py-0.5 rounded" 
                               :class="calculateAccuracy(result) > 80 ? 'text-emerald-300 bg-emerald-900/30' : calculateAccuracy(result) > 60 ? 'text-yellow-300 bg-yellow-900/30' : 'text-red-300 bg-red-900/30'">
                            {{ calculateAccuracy(result).toFixed(1) }}%
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <!-- Scroll to top button -->
                  <div v-if="vehicleResults.length > 3" class="sticky bottom-0 pt-1 sm:pt-2">
                    <button 
                      @click="scrollToTop"
                      class="w-full bg-slate-700 hover:bg-slate-600 text-slate-300 hover:text-slate-100 py-1 sm:py-1.5 px-2 sm:px-3 rounded-lg text-xs font-medium transition-colors duration-200"
                    >
                      ↑ Scroll to Top
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
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

  </div>
</template>

<style scoped>
/* Simulation Time Display - Floating above network */
.simulation-time-display {
  @apply absolute top-4 left-1/2 transform -translate-x-1/2 z-10;
  @apply bg-white bg-opacity-90 px-4 py-2 rounded-lg shadow-lg;
  @apply text-2xl font-semibold text-gray-800 text-center;
  @apply border border-gray-200;
}

/* Responsive font scaling based on screen resolution */
@media (max-width: 1536px) {
  .simulation-time-display {
    @apply text-xl;
  }
}

@media (max-width: 1280px) {
  .simulation-time-display {
    @apply text-lg;
  }
}

@media (max-width: 1024px) {
  .simulation-time-display {
    @apply text-base;
  }
}

@media (max-width: 768px) {
  .simulation-time-display {
    @apply text-sm px-3 py-1.5;
  }
}

@media (max-width: 640px) {
  .simulation-time-display {
    @apply text-xs px-2 py-1;
  }
}

/* Portrait mode adjustments - Move to top left */
@media (orientation: portrait) and (max-width: 1366px) {
  .simulation-time-display {
    @apply text-sm px-3 py-1.5 top-2 left-2;
    transform: none;
  }
}

@media (orientation: portrait) and (max-width: 1024px) {
  .simulation-time-display {
    @apply text-xs px-2 py-1 top-1 left-1;
    transform: none;
  }
}

@media (orientation: portrait) and (max-width: 768px) {
  .simulation-time-display {
    @apply text-xs px-2 py-1 top-1 left-1;
    transform: none;
  }
}
</style>

<script>
import apiService from '../services/api.js'
import jsPDF from 'jspdf'

export default {
  name: 'SimDemoPage',
  data() {
    return {
      // Responsive state management
      viewport: {
        width: 0,
        height: 0,
        isMobile: false,
        isTablet: false,
        isDesktop: false,
        isUltraWide: false,
        orientation: 'portrait',
        pixelRatio: 1,
        isRetina: false
      },
      
      // Responsive breakpoints
      breakpoints: {
        xs: 320,
        sm: 640,
        md: 768,
        lg: 1024,
        xl: 1280,
        '2xl': 1536,
        '3xl': 1920,
        '4xl': 2560,
        '5xl': 3440,
        '6xl': 3840
      },
      
      // Map-related data
      mousePosition: { x: 0, y: 0 },
      startPoint: null,
      destinationPoint: null,
      isJourneyRunning: false,
      
      // Network visualization
      networkData: null,
      networkBounds: null,
      svgViewBox: "0 0 1000 1000",
      
      // Simulation playback
      isSimulationPlaying: false,
      simulationTime: 0,
      simulationStatus: {
        vehicles: 0,
        vehicles_in_route: 0,
        trips_added: 0,
        current_step: 0
      },
      
      // Vehicle management
      activeVehicles: [],
      vehicleUpdateInterval: null,
      
      // Legend visibility
      showLegend: true,
      legendCollapsed: false,
      legendAutoFoldTimer: null,
      
      // Finished vehicle overlay
      showFinishedVehicleOverlay: false,
      finishedVehicleMessage: '',
      finishedVehicleTimer: null,
      shownFinishedVehicles: new Set(), // Track which vehicles we've already shown
      
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
    // Responsive utilities
    updateViewport() {
      this.viewport.width = window.innerWidth
      this.viewport.height = window.innerHeight
      this.viewport.pixelRatio = window.devicePixelRatio || 1
      this.viewport.isRetina = this.viewport.pixelRatio > 1
      this.viewport.orientation = window.innerHeight > window.innerWidth ? 'portrait' : 'landscape'
      
      // Determine device type
      this.viewport.isMobile = this.viewport.width < this.breakpoints.md
      this.viewport.isTablet = this.viewport.width >= this.breakpoints.md && this.viewport.width < this.breakpoints.lg
      this.viewport.isDesktop = this.viewport.width >= this.breakpoints.lg && this.viewport.width < this.breakpoints['3xl']
      this.viewport.isUltraWide = this.viewport.width >= this.breakpoints['3xl']
    },
    
    // Get current breakpoint
    getCurrentBreakpoint() {
      const width = this.viewport.width
      if (width < this.breakpoints.xs) return 'xs'
      if (width < this.breakpoints.sm) return 'sm'
      if (width < this.breakpoints.md) return 'md'
      if (width < this.breakpoints.lg) return 'lg'
      if (width < this.breakpoints.xl) return 'xl'
      if (width < this.breakpoints['2xl']) return '2xl'
      if (width < this.breakpoints['3xl']) return '3xl'
      if (width < this.breakpoints['4xl']) return '4xl'
      if (width < this.breakpoints['5xl']) return '5xl'
      return '6xl'
    },
    
    // Responsive class helper
    getResponsiveClasses(baseClasses = '') {
      const breakpoint = this.getCurrentBreakpoint()
      const orientation = this.viewport.orientation
      const isRetina = this.viewport.isRetina
      
      let classes = baseClasses
      
      // Add breakpoint-specific classes
      classes += ` breakpoint-${breakpoint}`
      
      // Add orientation classes
      classes += ` orientation-${orientation}`
      
      // Add retina class
      if (isRetina) classes += ' retina'
      
      // Add device type classes
      if (this.viewport.isMobile) classes += ' mobile'
      if (this.viewport.isTablet) classes += ' tablet'
      if (this.viewport.isDesktop) classes += ' desktop'
      if (this.viewport.isUltraWide) classes += ' ultra-wide'
      
      return classes
    },
    
    // Handle resize with debouncing
    handleResize() {
      clearTimeout(this.resizeTimeout)
      this.resizeTimeout = setTimeout(() => {
        this.updateViewport()
        this.onViewportChange()
      }, 100)
    },
    
    // Called when viewport changes
    onViewportChange() {
      // Override this method in your components for custom responsive behavior
      console.log('Viewport changed:', this.viewport)
    },
    
    // Map methods
    async loadNetworkData() {
      try {
        console.log('🔄 Loading network data...')
        const response = await apiService.getNetworkData()
        
        if (response) {
          this.networkData = response
          this.networkBounds = response.bounds
          
          // Calculate optimal viewBox with asymmetric padding
          const paddingX = 0.02 // 2% padding on sides
          const paddingY = 0.01 // 1% padding on top and bottom
          const width = this.networkBounds.max_x - this.networkBounds.min_x
          const height = this.networkBounds.max_y - this.networkBounds.min_y
          const paddingXValue = width * paddingX
          const paddingYValue = height * paddingY
          
          this.svgViewBox = `${this.networkBounds.min_x - paddingXValue} ${this.networkBounds.min_y - paddingYValue} ${width + 2 * paddingXValue} ${height + 2 * paddingYValue}`
          
          console.log('✅ Network data loaded successfully:', {
            edges: this.networkData?.edges?.length || 0,
            junctions: this.networkData?.junctions?.length || 0,
            bounds: this.networkBounds,
            viewBox: this.svgViewBox
          })
        } else {
          console.error('❌ Failed to load network data: No response')
        }
      } catch (error) {
        console.error('❌ Error loading network data:', error)
        this.networkData = null
      }
    },
    
    
    getJunctionPosition(junctionId) {
      if (!this.networkData || !this.networkData.junctions) return null
      return this.networkData.junctions.find(j => j.id === junctionId)
    },
    
    isExpressEdge(edge) {
      return edge.id && (edge.id.startsWith('-E') || edge.id.startsWith('E'))
    },
    
    getPathData(shapePoints) {
      if (!shapePoints || shapePoints.length < 2) return ''
      
      let pathData = `M ${shapePoints[0][0]} ${shapePoints[0][1]}`
      for (let i = 1; i < shapePoints.length; i++) {
        pathData += ` L ${shapePoints[i][0]} ${shapePoints[i][1]}`
      }
      return pathData
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
          return '#dc2626'  // Red
        default:
          return '#6b7280'  // Gray
      }
    },
    

    getVehicleTransform(vehicle) {
      // Simple positioning - no rotation
      return `translate(${vehicle.x}, ${vehicle.y})`
    },

    
    handleMapClick(event) {
      // Only show overlay if simulation is running and clicking on empty SVG area
      if ((this.isSimulationPlaying || this.isJourneyRunning) && event.target.tagName === 'svg') {
        console.log('🚗 Simulation running - showing overlay on map click')
      }
    },
    
    handleMouseMove(event) {
      const svg = this.$refs.networkSvg
      if (!svg) return

      const rect = svg.getBoundingClientRect()
      this.mousePosition = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      }
    },
    
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
          target: svg
        }
        
        this.handleMapClick(clickEvent)
      }
    },
    
    handleEdgeClick(edge, event) {
      console.log('🔍 handleEdgeClick called:', {
        edgeId: edge.id,
        eventType: event.type,
        isSimulationPlaying: this.isSimulationPlaying,
        isJourneyRunning: this.isJourneyRunning
      })
      
      event.preventDefault()
      event.stopPropagation()
      
      // Calculate the exact middle position of the edge
      let x, y
      
      if (edge.shape_points && edge.shape_points.length > 0) {
        const totalLength = edge.shape_points.length
        const midIndex = Math.floor(totalLength / 2)
        
        if (totalLength % 2 === 0 && midIndex > 0) {
          const point1 = edge.shape_points[midIndex - 1]
          const point2 = edge.shape_points[midIndex]
          x = (point1[0] + point2[0]) / 2
          y = (point1[1] + point2[1]) / 2
        } else {
          x = edge.shape_points[midIndex][0]
          y = edge.shape_points[midIndex][1]
        }
      } else {
        const fromJunction = this.getJunctionPosition(edge.from_junction)
        const toJunction = this.getJunctionPosition(edge.to_junction)
        if (fromJunction && toJunction) {
          x = (fromJunction.x + toJunction.x) / 2
          y = (fromJunction.y + toJunction.y) / 2
        } else {
          x = 0
          y = 0
        }
      }
      
      if (!this.startPoint) {
        this.startPoint = { id: edge.id, x, y }
        console.log('📍 Start point set:', this.startPoint)
      } else if (!this.destinationPoint) {
        this.destinationPoint = { id: edge.id, x, y }
        console.log('📍 Destination point set:', this.destinationPoint)
        this.calculateAndDisplayRoute()
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
    
    generateRoutePath(edges) {
      if (!edges || edges.length === 0) return ''
      
      let pathData = ''
      let isFirst = true
      
      for (const edgeId of edges) {
        const edge = this.networkData.edges.find(e => e.id === edgeId)
        if (!edge) continue
        
        if (edge.shape_points && edge.shape_points.length > 0) {
          if (isFirst) {
            pathData = `M ${edge.shape_points[0][0]} ${edge.shape_points[0][1]}`
            isFirst = false
          }
          for (let i = 1; i < edge.shape_points.length; i++) {
            pathData += ` L ${edge.shape_points[i][0]} ${edge.shape_points[i][1]}`
          }
        } else {
          const fromJunction = this.getJunctionPosition(edge.from_junction)
          const toJunction = this.getJunctionPosition(edge.to_junction)
          if (fromJunction && toJunction) {
            if (isFirst) {
              pathData = `M ${fromJunction.x} ${fromJunction.y}`
              isFirst = false
            }
            pathData += ` L ${toJunction.x} ${toJunction.y}`
          }
        }
      }
      
      return pathData
    },
    
    resetPoints() {
      this.startPoint = null
      this.destinationPoint = null
      this.routePath = null
      this.routeEdges = null
      this.routeDistance = null
      this.routeDuration = null
      console.log('🔄 Points reset')
    },
    
    handleMainButtonClick() {
        this.startJourney()
    },
    
    getMainButtonClass() {
      return 'bg-blue-600 hover:bg-blue-700'
    },
    
    getMainButtonText() {
      return '▶️ Start Journey'
    },
    
    async startJourney() {
      if (!this.startPoint || !this.destinationPoint) return
      
      console.log('🚀 Starting journey from', this.startPoint, 'to', this.destinationPoint)
      
      try {
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
        
        console.log('🛣️ Route edges available:', this.routeEdges)
        
        // Convert Proxy array to plain JavaScript array
        const routeEdgesArray = Array.from(this.routeEdges)
        console.log('🛣️ Route edges array:', routeEdgesArray)
        
        // Call the regular API to add vehicle to simulation
        console.log('🚀 Calling apiService.startJourney with:', {
          startEdge: this.startPoint.id,
          endEdge: this.destinationPoint.id,
          routeEdges: routeEdgesArray,
          routeEdgesType: typeof routeEdgesArray,
          routeEdgesLength: routeEdgesArray.length,
          routeEdgesIsArray: Array.isArray(routeEdgesArray)
        })
        
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
        
        // Try to get more detailed error information
        let errorMessage = 'Failed to start journey. Please try again.'
        if (error.response) {
          // Server responded with error status
          console.error('Server error response:', error.response.data)
          errorMessage = `Server error: ${error.response.status} - ${error.response.data?.detail || error.response.data?.message || 'Unknown error'}`
        } else if (error.request) {
          // Request was made but no response received
          console.error('No response received:', error.request)
          errorMessage = 'No response from server. Please check if the backend is running.'
        } else {
          // Something else happened
          console.error('Request setup error:', error.message)
          errorMessage = `Request error: ${error.message}`
        }
        
        alert(errorMessage)
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
      
      // Increment total journey count for new journey
      this.totalJourneyCount++
      
      // Add to beginning of array (most recent first)
      this.vehicleResults.unshift(result)
      
      // Keep only maxResults
      if (this.vehicleResults.length > this.maxResults) {
        this.vehicleResults = this.vehicleResults.slice(0, this.maxResults)
      }
      
      console.log('📊 Vehicle result added:', result)
      console.log('📊 Total journey count incremented to:', this.totalJourneyCount)
    },
    
    
    startSimulationPlayback() {
      this.isSimulationPlaying = true
      this.simulationUpdateInterval = setInterval(() => {
        this.updateSimulationStatus()
      }, 1000)
    },
    
    stopSimulationPlayback() {
      this.isSimulationPlaying = false
      if (this.simulationUpdateInterval) {
        clearInterval(this.simulationUpdateInterval)
        this.simulationUpdateInterval = null
      }
    },
    
    async updateSimulationStatus() {
      try {
        const response = await apiService.getSimulationStatus()
        if (response.success) {
          this.simulationStatus = response.data
        }
      } catch (error) {
        console.error('❌ Error updating simulation status:', error)
      }
    },
    
    // Vehicle management methods
    startVehicleUpdates() {
      this.vehicleUpdateInterval = setInterval(() => {
        this.loadActiveVehicles()
        // Also update simulation status to keep running time current
        this.loadSimulationStatus()
        // Always check for finished vehicles regardless of simulation state
        this.checkFinishedVehicles()
      }, 1000) // Update every second
    },
    
    stopVehicleUpdates() {
      if (this.vehicleUpdateInterval) {
        clearInterval(this.vehicleUpdateInterval)
        this.vehicleUpdateInterval = null
      }
    },
    
    async loadActiveVehicles() {
      try {
        const response = await apiService.getActiveVehicles()
        
        if (response && response.vehicles && Array.isArray(response.vehicles)) {
          this.activeVehicles = response.vehicles
        } else if (response && Array.isArray(response)) {
          this.activeVehicles = response
        } else {
          console.log('📊 No active vehicles found')
          this.activeVehicles = []
        }
      } catch (error) {
        console.error('❌ Error loading active vehicles:', error)
        this.activeVehicles = []
      }
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
    
    getInstructionText() {
      if (!this.startPoint) {
        return "Click on a road"
      } else if (!this.destinationPoint) {
        return "Click on a road"
      } else if (!this.isJourneyRunning) {
        return "Click 'Start Journey'"
      } else {
        return "Traveling to destination..."
      }
    },
    
    async loadSimulationStatus() {
      try {
        // console.log('🔄 Loading simulation status...')
        const response = await apiService.getSimulationStatus()
        
        if (response) {
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
          
          // console.log('✅ Simulation status loaded:', this.simulationStatus)
        } else {
          console.log('📊 No simulation status available')
          this.simulationStatus = {
            vehicles: 0,
            vehicles_in_route: 0,
            trips_added: 0,
            current_step: 0
          }
        }
      } catch (error) {
        console.error('❌ Failed to load simulation status:', error)
        this.simulationStatus = {
          vehicles: 0,
          vehicles_in_route: 0,
          trips_added: 0,
          current_step: 0
        }
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
    
    async updateVehicleResult(vehicleId, endTime, actualDuration, predictedDuration) {
      const result = this.vehicleResults.find(r => r.vehicle_id === vehicleId)
      if (result) {
        result.status = 'finished'
        result.end_time = endTime
        result.end_time_string = this.formatTime(endTime)
        result.actual_duration = actualDuration
        
        const absoluteError = Math.abs(predictedDuration - actualDuration)
        const accuracy = actualDuration > 0 ? Math.max(0, 100 - (absoluteError / actualDuration) * 100) : 0
        
        result.absolute_error = absoluteError
        result.accuracy = accuracy
        
        console.log('📊 Vehicle result updated:', result)
        
        // Set journey as no longer running to show buttons again
        this.isJourneyRunning = false
        
        try {
          await this.saveJourneyToDatabase(result)
          await this.loadJourneyStatistics()
          // Refresh recent journeys from database after saving
          await this.loadRecentJourneysFromDB()
        } catch (error) {
          console.error('❌ Error saving journey to database:', error)
        }
      } else {
        console.warn('⚠️ Vehicle result not found for vehicle ID:', vehicleId)
      }
    },
    
    async saveJourneyToDatabase(journeyResult) {
      try {
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
        return response
      } catch (error) {
        console.error('❌ Error saving journey to database:', error)
        throw error
      }
    },
    
    async openPlotWindow(plotType) {
      console.log('📊 Opening plot window for:', plotType)
      this.currentPlotType = plotType
      
      try {
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
          const response = await apiService.getDistanceVsMaePlotImage()
          if (response.success) {
            this.plotImage = response.image
            this.currentPlotData = { total_points: response.total_points }
            this.showPlotModal = true
          } else {
            console.error('Failed to load plot image:', response)
            alert('Failed to load plot image. Please try again.')
          }
        } else if (plotType === 'mae-by-time') {
          const response = await apiService.getMaeByTimePlotImage()
          if (response.success) {
            this.plotImage = response.image
            this.currentPlotData = { total_points: response.total_points }
            this.showPlotModal = true
          } else {
            console.error('Failed to load plot image:', response)
            alert('Failed to load plot image. Please try again.')
          }
        } else if (plotType === 'duration-histogram-short' || plotType === 'duration-histogram-medium' || plotType === 'duration-histogram-long') {
          const category = plotType.split('-')[2]
          const response = await apiService.getDurationHistogramPlotImage(category)
          if (response.success) {
            this.plotImage = response.image
            this.currentPlotData = { total_points: response.total_points, total_journeys: response.total_journeys }
            this.showPlotModal = true
          } else {
            console.error('Failed to load plot image:', response)
            alert('Failed to load plot image. Please try again.')
          }
        } else if (plotType === 'distance-histogram-short' || plotType === 'distance-histogram-medium' || plotType === 'distance-histogram-long') {
          const category = plotType.split('-')[2]
          const response = await apiService.getDistanceHistogramPlotImage(category)
          if (response.success) {
            this.plotImage = response.image
            this.currentPlotData = { total_points: response.total_points, total_journeys: response.total_journeys }
            this.showPlotModal = true
          } else {
            console.error('Failed to load plot image:', response)
            alert('Failed to load plot image. Please try again.')
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
      if (this.chartInstance && window.Plotly) {
        try {
          window.Plotly.purge(this.chartInstance)
        } catch (error) {
          console.warn('Error purging Plotly chart:', error)
        }
        this.chartInstance = null
      }
      
      if (this.resizeHandler) {
        window.removeEventListener('resize', this.resizeHandler)
        this.resizeHandler = null
      }
    },
    
    formatTime(seconds) {
      let totalSeconds = seconds
      
      if (typeof seconds === 'string') {
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
      
      if (isNaN(totalSeconds) || totalSeconds < 0) {
        totalSeconds = 0
      }
      
      const hours = Math.floor(totalSeconds / 3600)
      const minutes = Math.floor((totalSeconds % 3600) / 60)
      const secs = Math.floor(totalSeconds % 60)
      
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    },
    
    getInstructionText() {
      if (!this.startPoint) {
        return 'Click on any road to set your starting point'
      } else if (!this.destinationPoint) {
        return 'Click on another road to set your destination'
      } else if (!this.isJourneyRunning) {
        return 'Click "Start Journey" to begin your trip'
      } else {
        return 'Your journey is in progress. Watch the vehicle move!'
      }
    },

    // Results section methods
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

    scrollToTop() {
      const resultsContainer = this.$el.querySelector('.overflow-y-auto')
      if (resultsContainer) {
        resultsContainer.scrollTo({ top: 0, behavior: 'smooth' })
      }
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
  async mounted() {
    // Initialize viewport
    this.updateViewport()
    
    // Add resize listener
    window.addEventListener('resize', this.handleResize)
    window.addEventListener('orientationchange', this.handleResize)
    
    // Add visibility change listener for performance
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) {
        this.updateViewport()
      }
    })
    
    // Load network data for the map
    this.loadNetworkData()
    
    // Load simulation status and journey statistics
    this.loadSimulationStatus()
    this.loadJourneyStatistics()
    
    // Load recent journeys from database (must await so list is filled before UI settles)
    await this.loadRecentJourneysFromDB()
    
    // Clear any old finished vehicles to prevent showing stale data on page refresh
    try {
      await apiService.clearFinishedVehicles()
      console.log('🧹 Cleared old finished vehicles on page load')
    } catch (error) {
      console.warn('⚠️ Could not clear finished vehicles:', error)
    }
    
    // Load vehicles immediately and start updates
    this.loadActiveVehicles()
    this.startVehicleUpdates()
    
    // Start legend auto-fold timer
    this.startLegendAutoFold()
  },
  beforeUnmount() {
    // Clean up listeners
    window.removeEventListener('resize', this.handleResize)
    window.removeEventListener('orientationchange', this.handleResize)
    document.removeEventListener('visibilitychange', this.updateViewport)
    
    if (this.resizeTimeout) {
      clearTimeout(this.resizeTimeout)
    }
    
    // Clean up legend auto-fold timer
    if (this.legendAutoFoldTimer) {
      clearTimeout(this.legendAutoFoldTimer)
      this.legendAutoFoldTimer = null
    }
    
    // Clean up finished vehicle timer
    if (this.finishedVehicleTimer) {
      clearTimeout(this.finishedVehicleTimer)
      this.finishedVehicleTimer = null
    }
    
    // Clean up intervals
    this.stopVehicleUpdates()
    this.stopSimulationPlayback()
    
    // Clean up chart instance
    if (this.chartInstance && window.Plotly) {
      try {
        window.Plotly.purge(this.chartInstance)
      } catch (error) {
        console.warn('Error purging Plotly chart:', error)
      }
      this.chartInstance = null
    }
    
    if (this.resizeHandler) {
      window.removeEventListener('resize', this.resizeHandler)
      this.resizeHandler = null
    }
  }
}
</script>

<style scoped>
/* ===== APP CONTAINER ===== */
.app-container {
  height: 100vh;
  width: 100vw;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  margin: 0;
  padding: 0;
}

/* Prevent scroll bars for this component */

/* Responsive utilities and edge case handling */

/* ===== HEADER GRID LAYOUT ===== */
.header-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  align-items: center;
  height: 2.1rem;
  width: 100%;
  max-width: 100%;
}

@media (min-width: 320px) {
  .header-grid {
    height: 2.45rem;
  }
}

@media (min-width: 640px) {
  .header-grid {
    height: 2.8rem;
  }
}

@media (min-width: 768px) {
  .header-grid {
    height: 3.15rem;
  }
}

@media (min-width: 1024px) {
  .header-grid {
    height: 3.5rem;
  }
}

/* ===== HEADER LEFT SIDE ===== */
.header-left {
  display: flex;
  align-items: center;
  margin: 0;
  padding: 0;
}

.header-logo {
  font-size: 0.75rem;
  font-weight: bold;
  color: white;
  text-decoration: none;
  white-space: nowrap;
  transition: color 0.2s ease-in-out;
}

.header-logo:hover {
  color: #93c5fd;
}

@media (min-width: 320px) {
  .header-logo {
    font-size: 0.875rem;
  }
}

@media (min-width: 640px) {
  .header-logo {
    font-size: 1.125rem;
  }
}

@media (min-width: 768px) {
  .header-logo {
    font-size: 1.25rem;
  }
}

@media (min-width: 1024px) {
  .header-logo {
    font-size: 1.05rem;
  }
}

@media (min-width: 1280px) {
  .header-logo {
    font-size: 1.3125rem;
  }
}

/* ===== HEADER RIGHT SIDE ===== */
.header-right {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  margin: 0;
  padding: 0;
}

.header-subtitle {
  font-size: 0.875rem;
  font-weight: 500;
  color: white;
  text-align: right;
  line-height: 1.25;
  white-space: nowrap;
}

/* Mobile text - show on small screens */
.header-mobile-text {
  display: inline !important;
}

.header-desktop-text {
  display: none !important;
}

/* Desktop text - show on larger screens */
@media (min-width:720px) {
  .header-mobile-text {
    display: none !important;
  }
  
  .header-desktop-text {
    display: inline !important;
  }
}

@media (min-width: 320px) {
  .header-subtitle {
    font-size: 1rem;
  }
}

@media (min-width: 640px) {
  .header-subtitle {
    font-size: 1.125rem;
  }
}

@media (min-width: 768px) {
  .header-subtitle {
    font-size: 1.25rem;
  }
}

@media (min-width: 1024px) {
  .header-subtitle {
    font-size: 1.05rem;
  }
}

@media (min-width: 1280px) {
  .header-subtitle {
    font-size: 1.3125rem;
  }
}

/* Handle very small screens with text truncation */
@media (max-width: 480px) {
  nav .flex {
    gap: 0.25rem;
  }
  
  nav span {
    font-size: 0.625rem !important;
    line-height: 1rem !important;
  }
}

/* Ensure right text is visible */
nav span {
  color: white !important;
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
}

/* Ensure proper text wrapping and overflow handling */
.responsive-text {
  word-wrap: break-word;
  overflow-wrap: break-word;
  hyphens: auto;
}

/* Handle very small screens with horizontal scrolling prevention */
@media (max-width: 320px) {
  .container {
    min-width: 320px;
  }
}

/* Ultra-wide screen optimizations */
@media (min-width: 2560px) {
  .container {
    max-width: 90vw;
    margin: 0 auto;
  }
}

/* High DPI display optimizations */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
  .text {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
}

/* Landscape mobile optimizations */
@media (orientation: landscape) and (max-height: 500px) {
  .nav-height {
    height: 3rem;
  }
}

/* Portrait mobile optimizations */
@media (orientation: portrait) and (max-width: 480px) {
  .nav-height {
    height: 3.5rem;
  }
}

/* Ensure proper touch targets on mobile */
@media (max-width: 768px) {
  .touch-target {
    min-height: 44px;
    min-width: 44px;
  }
}

/* Smooth transitions for responsive changes */
* {
  transition: font-size 0.2s ease-in-out, padding 0.2s ease-in-out, margin 0.2s ease-in-out;
}

/* Prevent horizontal scroll on all devices */
html, body {
  overflow-x: hidden;
  max-width: 100vw;
}

/* Container queries support (when available) */
@container (min-width: 320px) {
  .responsive-content {
    font-size: 0.75rem;
  }
}

@container (min-width: 640px) {
  .responsive-content {
    font-size: 0.875rem;
  }
}

@container (min-width: 1024px) {
  .responsive-content {
    font-size: 1rem;
  }
}

/* Print styles */
@media print {
  .no-print {
    display: none !important;
  }
  
  .print-friendly {
    color: black !important;
    background: white !important;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .high-contrast {
    border: 2px solid;
  }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
  .dark-mode {
    background-color: #1a1a1a;
    color: #ffffff;
  }
}

/* ===== MAIN CONTENT LAYOUT ===== */
.main-content {
  flex: 1;
  width: 100%;
  padding: 0;
  overflow: hidden;
  box-sizing: border-box;
  position: relative;
}

/* Portrait mode - Allow scrolling */
@media (orientation: portrait) and (max-width: 1366px) {
  .main-content {
    overflow-y: auto;
    height: 100vh;
  }
}

/* ===== LANDSCAPE MESSAGE (Desktop/Landscape) ===== */
.landscape-message {
  display: none;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 100%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  text-align: center;
  padding: 2rem;
  box-sizing: border-box;
}

.landscape-content {
  max-width: 400px;
}

.phone-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
  animation: rotate 2s ease-in-out infinite;
}

.portrait-content h2 {
  font-size: 2rem;
  margin: 0 0 1rem 0;
  font-weight: bold;
}

.portrait-content p {
  font-size: 1.1rem;
  margin: 0 0 2rem 0;
  line-height: 1.5;
  opacity: 0.9;
}

.rotation-hint {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.arrow {
  font-size: 2rem;
  animation: bounce 1s ease-in-out infinite;
}

.rotation-hint span {
  font-size: 1rem;
  opacity: 0.8;
}

/* ===== SIMULATION LAYOUT (Portrait Only) ===== */
.simulation-layout {
  display: block;
  height: 100%;
  width: 100%;
}

.content-grid {
  display: grid;
  grid-template-columns: 20% 60% 20%;
  gap: 0.25rem;
  height: 100%;
  width: 100%;
  padding: 0.25rem;
  box-sizing: border-box;
}

/* Portrait layout for mobile, tablets, and iPad Pro */
@media (orientation: portrait) and (max-width: 1366px) {
  .content-grid {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    min-height: 150vh; /* Allow content to extend beyond viewport for larger results section */
    width: 100%;
    padding: 0.25rem;
    box-sizing: border-box;
  }
  
  /* Reorder sections for portrait */
  .map-section {
    order: 1;
    position: relative;
    z-index: 1;
    background-color: #dbeafe !important;
    border: 2px solid #1d4ed8 !important;
    height: 40vh;
    min-height: 300px;
  }
  
  .analysis-section {
    order: 2;
    position: relative;
    z-index: 1;
    overflow-y: auto;
    background-color: #f3f4f6 !important;
    border: 2px solid #3b82f6 !important;
    height: 60vh;
    min-height: 400px;
  }
  
  .results-section {
    order: 3;
    position: relative;
    z-index: 1;
    overflow-y: auto;
    background-color: #f0fdf4 !important;
    border: 2px solid #22c55e !important;
    height: 50vh;
    min-height: 400px;
  }
}

/* ===== RESPONSIVE BEHAVIOR ===== */
/* Desktop/Laptop - Always show simulation regardless of orientation */
@media (min-width: 1024px) {
  .landscape-message {
    display: none;
  }
  
  .simulation-layout {
    display: block;
  }
}

/* Landscape-only overlay - hide in portrait mode */
.landscape-only {
  display: none;
}

@media (min-width: 640px) and (orientation: landscape) {
  .landscape-only {
    display: block;
  }
}

/* Desktop always shows landscape overlay regardless of orientation */
@media (min-width: 1024px) {
  .landscape-only {
    display: block;
  }
}


/* Mobile and Tablets - Portrait only simulation */
/* Show simulation layout on mobile and portrait tablets */
@media (max-width: 1023px) and (orientation: portrait) {
  .landscape-message {
    display: none;
  }
  
  .simulation-layout {
    display: block;
  }
}

/* Show landscape message on mobile and tablets in landscape */
@media (max-width: 1023px) and (orientation: landscape) {
  .landscape-message {
    display: flex;
  }
  
  .simulation-layout {
    display: none;
  }
}

/* ===== ANIMATIONS ===== */
@keyframes rotate {
  0%, 100% { transform: rotate(0deg); }
  50% { transform: rotate(90deg); }
}

@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

/* ===== ANALYSIS SECTION (20%) ===== */
.analysis-section {
  background-color: #f3f4f6;
  border: 1px solid #e5e7eb;
  border-radius: 0.25rem;
  padding: 0.25rem;
  overflow-y: auto;
  box-sizing: border-box;
}

/* Portrait mode - Make analysis section much taller */
@media (orientation: portrait) and (max-width: 1366px) {
  .analysis-section {
    min-height: 70vh; /* Much taller in portrait mode */
    max-height: 80vh; /* Prevent it from being too tall */
    position: relative;
    z-index: 2;
    overflow-y: auto;
  }
  
  /* Make results section much taller in portrait mode */
  .results-section {
    min-height: 60vh; /* Much taller for results */
    max-height: 70vh; /* Prevent it from being too tall */
  }
}

/* ===== MAP SECTION (60%) ===== */
.map-section {
  background-color: #dbeafe;
  border: 1px solid #3b82f6;
  border-radius: 0.25rem;
  padding: 0.25rem;
  overflow: hidden;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
}


.map-container {
  position: relative;
  flex: 1;
  background-color: #f1f5f9;
  overflow: hidden;
}

.map-container {
  width: 100%;
  height: 100%;
  background: #f1f5f9;
}


/* Map elements - Exact CSS from original DemoPage */
.road-path {
  pointer-events: stroke;
  transition: all 0.2s ease;
}

/* Ensure touch events work on mobile and tablets */
@media (max-width: 1366px) {
  .road-path,
  path[data-road-name],
  line[data-road-name] {
    pointer-events: all;
    touch-action: manipulation;
  }
}

/* Control buttons always stack vertically */
.absolute.top-2.right-2 > div {
  flex-direction: column !important;
  align-items: stretch !important;
}

/* Responsive button sizing for different screen sizes */
@media (max-width: 480px) {
  .absolute.top-2.right-2 {
    top: 0.25rem;
    right: 0.25rem;
  }
  
  .absolute.top-2.right-2 button {
    padding: 0.25rem 0.5rem;
    font-size: 0.625rem;
    border-radius: 0.25rem;
  }
}

@media (max-width: 320px) {
  .absolute.top-2.right-2 {
    top: 0.125rem;
    right: 0.125rem;
  }
  
  .absolute.top-2.right-2 button {
    padding: 0.125rem 0.25rem;
    font-size: 0.5rem;
    border-radius: 0.125rem;
  }
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

.start-marker,
.dest-marker {
  cursor: pointer;
  transition: r 0.2s ease;
  filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
}

.start-marker:hover,
.dest-marker:hover {
  r: 15;
}

.marker-text {
  pointer-events: none;
  user-select: none;
}

.route-path {
  stroke-dasharray: 5,5;
  animation: dash 1s linear infinite;
  stroke-linecap: round;
  stroke-linejoin: round;
}

@keyframes dash {
  to {
    stroke-dashoffset: -10;
  }
}

.vehicle-marker {
  cursor: pointer;
  transition: r 0.2s ease;
  opacity: 0.8;
}

.vehicle-marker:hover {
  r: 45;
}

/* Road tooltip animation */
.road-tooltip {
  animation: fadeIn 0.2s ease-in-out;
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

/* Legend animation */
.animate-fade-in {
  animation: fadeIn 0.3s ease-in-out;
}

/* Hide scrollbar for webkit browsers */
.hide-scrollbar::-webkit-scrollbar {
  display: none;
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

/* Legend animations */
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

.animate-fade-in {
  animation: fadeIn 0.3s ease-in-out;
}

/* ===== RESULTS SECTION (20%) ===== */
.results-section {
  background-color: #f0fdf4;
  border: 1px solid #22c55e;
  border-radius: 0.25rem;
  padding: 0.25rem;
  overflow-y: auto;
  box-sizing: border-box;
}

/* ===== SECTION PLACEHOLDERS ===== */
.section-placeholder {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: #374151;
}

.section-placeholder h3 {
  margin: 0 0 0.5rem 0;
  font-size: 1.25rem;
  font-weight: bold;
}

.section-placeholder p {
  margin: 0;
  font-size: 0.875rem;
  color: #6b7280;
}
</style>
