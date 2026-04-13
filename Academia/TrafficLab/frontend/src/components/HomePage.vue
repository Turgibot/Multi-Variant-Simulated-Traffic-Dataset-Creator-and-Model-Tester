<template>
  <div class="w-full overflow-x-hidden">
    <!-- Hero Section -->
    <section id="home" class="bg-gradient-to-r from-blue-600 to-blue-800 text-white py-12 sm:py-16 lg:py-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h1 class="text-3xl sm:text-4xl lg:text-5xl font-bold mb-4 sm:mb-6">SmartTransportation Lab</h1>
        <p class="text-lg sm:text-xl mb-6 sm:mb-8 px-2">Advancing Traffic Simulation and ETA Prediction through AI</p>
        <router-link to="/sim-demo" class="bg-white text-blue-600 px-6 sm:px-8 py-2 sm:py-3 rounded-lg font-semibold hover:bg-blue-50 transition duration-300 inline-block text-sm sm:text-base">
          Try Live Demo
        </router-link>
      </div>
    </section>

    <!-- About Section -->
    <section id="about" class="py-12 sm:py-16 bg-gray-50">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 class="text-2xl sm:text-3xl font-bold text-center mb-8 sm:mb-12">About Our Lab</h2>
        <div class="grid md:grid-cols-2 gap-6 sm:gap-8">
          <div>
            <h3 class="text-xl sm:text-2xl font-semibold mb-3 sm:mb-4">Our Mission</h3>
            <p class="text-gray-600 leading-relaxed text-sm sm:text-base">
              The SmartTransportation Lab is dedicated to advancing traffic simulation technologies 
              and developing accurate ETA prediction models using cutting-edge machine learning techniques. 
              We combine SUMO traffic simulation with PyTorch-based neural networks to create 
              intelligent transportation solutions.
            </p>
          </div>
          <div>
            <h3 class="text-xl sm:text-2xl font-semibold mb-3 sm:mb-4">Research Focus</h3>
            <ul class="text-gray-600 space-y-1 sm:space-y-2 text-sm sm:text-base">
              <li>• ETA prediction models and algorithms</li>
              <li>• Advanced data analysis and visualization</li>
              <li>• Network analysis and graph theory applications</li>
              <li>• Smart city infrastructure and optimization</li>
              <li>• Traffic flow simulation and modeling</li>
              <li>• Machine learning for transportation systems</li>
              <li>• Real-time traffic data analysis</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <!-- Team Section -->
    <section id="team" class="py-12 sm:py-16 bg-gray-50">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 class="text-2xl sm:text-3xl font-bold text-center mb-4 sm:mb-6">Our Research Team</h2>
        <p class="text-center text-gray-600 mb-8 sm:mb-12 max-w-3xl mx-auto text-sm sm:text-base px-2">
          Led by Principal Investigator Nadav Voloch, our research team consists of 1 Principal Investigator and 3 Research Scientists, 
          bringing together expertise in network analysis, smart city applications, ETA prediction models, traffic simulation, 
          machine learning, and data analysis. We collaborate closely to advance the field of intelligent transportation systems 
          through innovative research and practical applications.
        </p>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4 sm:gap-6 lg:gap-8 max-w-4xl mx-auto">
          <div class="text-center bg-white rounded-lg shadow-md p-4 sm:p-6 hover:shadow-lg transition-shadow duration-300">
            <div class="w-32 h-32 rounded-full mx-auto mb-3 sm:mb-4 overflow-hidden border-4 border-blue-200">
              <img 
                src="/images/nadav-voloch.jpeg" 
                alt="Nadav Voloch - Principal Investigator"
                class="w-full h-full object-cover"
                @error="handleImageError"
                @load="handleImageLoad"
                v-show="showImage"
              />
              <div v-show="!showImage" class="w-full h-full bg-gradient-to-br from-blue-400 to-blue-600 rounded-full flex items-center justify-center">
                <span class="text-white text-2xl sm:text-4xl font-bold">NV</span>
              </div>
            </div>
            <h3 class="text-lg sm:text-xl font-semibold mb-1 sm:mb-2">Nadav Voloch</h3>
            <p class="text-blue-600 font-medium text-sm sm:text-base">Principal Investigator</p>
            <p class="text-gray-600 mt-1 sm:mt-2 text-xs sm:text-sm">Network Analysis & Smart City Applications</p>
            <p class="text-gray-500 text-xs mt-1">Lead Researcher</p>
            <div class="mt-2 sm:mt-3">
              <a href="https://www.linkedin.com/in/nadav-voloch-570a24254/" target="_blank" 
                 class="text-blue-500 hover:text-blue-700 text-xs sm:text-sm underline">
                LinkedIn Profile
              </a>
            </div>
          </div>
          <div class="text-center bg-white rounded-lg shadow-md p-4 sm:p-6 hover:shadow-lg transition-shadow duration-300">
            <div class="w-32 h-32 rounded-full mx-auto mb-3 sm:mb-4 overflow-hidden border-4 border-green-200">
              <img 
                src="/images/guy.jpeg" 
                alt="Guy Tordjman - Research Scientist"
                class="w-full h-full object-cover"
                @error="handleImageError2"
                @load="handleImageLoad2"
                v-show="showImage2"
              />
              <div v-show="!showImage2" class="w-full h-full bg-gradient-to-br from-green-400 to-green-600 rounded-full flex items-center justify-center">
                <span class="text-white text-2xl sm:text-4xl font-bold">GT</span>
              </div>
            </div>
            <h3 class="text-lg sm:text-xl font-semibold mb-1 sm:mb-2">Guy Tordjman</h3>
            <p class="text-green-600 font-medium text-sm sm:text-base">Research Scientist</p>
            <p class="text-gray-600 mt-1 sm:mt-2 text-xs sm:text-sm">ETA Models & Data Analysis</p>
            <p class="text-gray-500 text-xs mt-1">Senior Research Scientist</p>
            <div class="mt-2 sm:mt-3">
              <a href="https://www.linkedin.com/in/guy-tordjman/" target="_blank" 
                 class="text-green-500 hover:text-green-700 text-xs sm:text-sm underline">
                LinkedIn Profile
              </a>
            </div>
          </div>
          <div class="text-center bg-white rounded-lg shadow-md p-4 sm:p-6 hover:shadow-lg transition-shadow duration-300">
            <div class="w-32 h-32 rounded-full mx-auto mb-3 sm:mb-4 overflow-hidden border-4 border-purple-200">
              <img 
                src="/images/maor-meir-hajaj.jpeg" 
                alt="Maor Meir Hajaj - Research Scientist"
                class="w-full h-full object-cover"
                @error="handleImageError3"
                @load="handleImageLoad3"
                v-show="showImage3"
              />
              <div v-show="!showImage3" class="w-full h-full bg-gradient-to-br from-purple-400 to-purple-600 rounded-full flex items-center justify-center">
                <span class="text-white text-2xl sm:text-4xl font-bold">MM</span>
              </div>
            </div>
            <h3 class="text-lg sm:text-xl font-semibold mb-1 sm:mb-2">Maor Meir Hajaj</h3>
            <p class="text-purple-600 font-medium text-sm sm:text-base">Research Scientist</p>
            <p class="text-gray-600 mt-1 sm:mt-2 text-xs sm:text-sm">SUMO Integration & ETA Models</p>
            <p class="text-gray-500 text-xs mt-1">Research Scientist</p>
            <div class="mt-2 sm:mt-3">
              <a href="https://www.linkedin.com/in/maor-meir-hajaj/" target="_blank" 
                 class="text-purple-500 hover:text-purple-700 text-xs sm:text-sm underline">
                LinkedIn Profile
              </a>
            </div>
          </div>
          <div class="text-center bg-white rounded-lg shadow-md p-4 sm:p-6 hover:shadow-lg transition-shadow duration-300">
            <div class="w-32 h-32 rounded-full mx-auto mb-3 sm:mb-4 overflow-hidden border-4 border-orange-200">
              <img 
                src="/images/matan-shemesh.jpeg" 
                alt="Matan Shemesh - Research Scientist"
                class="w-full h-full object-cover"
                @error="handleImageError4"
                @load="handleImageLoad4"
                v-show="showImage4"
              />
              <div v-show="!showImage4" class="w-full h-full bg-gradient-to-br from-orange-400 to-orange-600 rounded-full flex items-center justify-center">
                <span class="text-white text-2xl sm:text-4xl font-bold">MS</span>
              </div>
            </div>
            <h3 class="text-lg sm:text-xl font-semibold mb-1 sm:mb-2">Matan Shemesh</h3>
            <p class="text-orange-600 font-medium text-sm sm:text-base">Research Scientist</p>
            <p class="text-gray-600 mt-1 sm:mt-2 text-xs sm:text-sm">Algorithm Development & Testing</p>
            <p class="text-gray-500 text-xs mt-1">Research Scientist</p>
            <div class="mt-2 sm:mt-3">
              <a href="https://www.linkedin.com/in/matanshemesh/" target="_blank" 
                 class="text-orange-500 hover:text-orange-700 text-xs sm:text-sm underline">
                LinkedIn Profile
              </a>
            </div>
          </div>
        </div>
        
        <!-- Team Stats -->
        <div class="mt-8 sm:mt-12 grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6 max-w-2xl mx-auto">
          <div class="text-center">
            <div class="text-2xl sm:text-3xl font-bold text-blue-600">4</div>
            <div class="text-xs sm:text-sm text-gray-600">Researchers</div>
          </div>
          <div class="text-center">
            <div class="text-2xl sm:text-3xl font-bold text-green-600">20+</div>
            <div class="text-xs sm:text-sm text-gray-600">Years Experience</div>
          </div>
          <div class="text-center">
            <div class="text-2xl sm:text-3xl font-bold text-purple-600">10+</div>
            <div class="text-xs sm:text-sm text-gray-600">Publications</div>
          </div>
          <div class="text-center">
            <div class="text-2xl sm:text-3xl font-bold text-orange-600">3</div>
            <div class="text-xs sm:text-sm text-gray-600">Active Projects</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Research Section -->
    <section id="research" class="py-12 sm:py-16">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 class="text-2xl sm:text-3xl font-bold text-center mb-8 sm:mb-12">Our Research</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
          <div class="bg-white rounded-lg shadow-md p-4 sm:p-6">
            <div class="text-3xl sm:text-4xl mb-3 sm:mb-4">🚗</div>
            <h3 class="text-lg sm:text-xl font-semibold mb-2 sm:mb-3">Traffic Simulation</h3>
            <p class="text-gray-600 mb-3 sm:mb-4 text-sm sm:text-base">
              Advanced SUMO-based traffic simulation for modeling complex urban traffic scenarios 
              and understanding traffic flow dynamics.
            </p>
            <ul class="text-xs sm:text-sm text-gray-500 space-y-1">
              <li>• Real-time traffic modeling</li>
              <li>• Multi-modal transportation</li>
              <li>• Intersection optimization</li>
            </ul>
          </div>
          
          <div class="bg-white rounded-lg shadow-md p-4 sm:p-6">
            <div class="text-3xl sm:text-4xl mb-3 sm:mb-4">🤖</div>
            <h3 class="text-lg sm:text-xl font-semibold mb-2 sm:mb-3">Machine Learning</h3>
            <p class="text-gray-600 mb-3 sm:mb-4 text-sm sm:text-base">
              Deep learning models for accurate ETA prediction using PyTorch, 
              processing real-time traffic data and historical patterns.
            </p>
            <ul class="text-xs sm:text-sm text-gray-500 space-y-1">
              <li>• Neural network architectures</li>
              <li>• Time series prediction</li>
              <li>• Data preprocessing pipelines</li>
            </ul>
          </div>
          
          <div class="bg-white rounded-lg shadow-md p-4 sm:p-6">
            <div class="text-3xl sm:text-4xl mb-3 sm:mb-4">📊</div>
            <h3 class="text-lg sm:text-xl font-semibold mb-2 sm:mb-3">Data Analytics</h3>
            <p class="text-gray-600 mb-3 sm:mb-4 text-sm sm:text-base">
              Comprehensive analysis of traffic patterns, performance metrics, 
              and predictive modeling for intelligent transportation systems.
            </p>
            <ul class="text-xs sm:text-sm text-gray-500 space-y-1">
              <li>• Statistical analysis</li>
              <li>• Performance evaluation</li>
              <li>• Visualization tools</li>
            </ul>
          </div>
        </div>
        
        <!-- Research Methodology -->
        <div class="mt-12 sm:mt-16 bg-gray-50 rounded-lg p-4 sm:p-6 lg:p-8">
          <h3 class="text-xl sm:text-2xl font-semibold mb-4 sm:mb-6 text-center">Research Methodology</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8">
            <div>
              <h4 class="text-base sm:text-lg font-semibold mb-2 sm:mb-3">Data Collection</h4>
              <p class="text-gray-600 mb-3 sm:mb-4 text-sm sm:text-base">
                We collect real-time traffic data from multiple sources including sensors, 
                GPS traces, and simulation outputs to build comprehensive datasets.
              </p>
            </div>
            <div>
              <h4 class="text-base sm:text-lg font-semibold mb-2 sm:mb-3">Model Development</h4>
              <p class="text-gray-600 mb-3 sm:mb-4 text-sm sm:text-base">
                Our models combine traditional traffic engineering principles with 
                modern machine learning techniques for robust ETA predictions.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Live Demo Section -->
    <section id="demo" class="py-12 sm:py-16 bg-blue-50">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 class="text-2xl sm:text-3xl font-bold mb-6 sm:mb-8">Live Demo</h2>
        <div class="bg-white rounded-lg shadow-lg p-6 sm:p-8 max-w-2xl mx-auto">
          <h3 class="text-xl sm:text-2xl font-semibold mb-3 sm:mb-4">SUMO-ETA Model Simulation</h3>
          <p class="text-gray-600 mb-4 sm:mb-6 text-sm sm:text-base">
            Experience our real-time traffic simulation combined with AI-powered ETA prediction. 
            Watch as our system processes live traffic data and provides accurate arrival time estimates.
          </p>
          <router-link to="/sim-demo" class="bg-blue-600 text-white px-6 sm:px-8 py-2 sm:py-3 rounded-lg font-semibold hover:bg-blue-700 transition duration-300 inline-block text-sm sm:text-base">
            Launch Interactive Demo
          </router-link>
        </div>
      </div>
    </section>

    <!-- Publications Section -->
    <section id="publications" class="py-12 sm:py-16">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 class="text-2xl sm:text-3xl font-bold text-center mb-8 sm:mb-12">Recent Publications</h2>
        <div class="space-y-4 sm:space-y-6">
          <div class="bg-white shadow-md rounded-lg p-4 sm:p-6 hover:shadow-lg transition-shadow duration-300">
            <h3 class="text-lg sm:text-xl font-semibold mb-2">Dynamic Route-Aware Graph Neural Networks for Accurate ETA Prediction</h3>
            <p class="text-gray-600 mb-2 text-sm sm:text-base">IEEE Intelligent Transportation Systems Transactions</p>
            <p class="text-xs sm:text-sm text-gray-500 mb-3 sm:mb-4">Authors: Guy Tordjman, Nadav Voloch</p>
            <a href="/publications/DTSRA-GNN_I3E_2025.pdf" target="_blank" 
               class="inline-flex items-center px-3 sm:px-4 py-1.5 sm:py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-300 text-xs sm:text-sm">
              <svg class="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
              Download PDF
            </a>
          </div>
          <div class="bg-white shadow-md rounded-lg p-4 sm:p-6 hover:shadow-lg transition-shadow duration-300">
            <h3 class="text-lg sm:text-xl font-semibold mb-2">Finding the fastest navigation route by real-time future traffic estimations</h3>
            <p class="text-gray-600 mb-2 text-sm sm:text-base">COMCAS 2021</p>
            <p class="text-xs sm:text-sm text-gray-500 mb-3 sm:mb-4">Authors: Nadav Voloch, Noa Voloch-Bloch</p>
            <a href="/publications/FastestETAWithFutureTrafficCOMCAS2021.pdf" target="_blank" 
               class="inline-flex items-center px-3 sm:px-4 py-1.5 sm:py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-300 text-xs sm:text-sm">
              <svg class="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
              Download PDF
            </a>
          </div>
          <div class="bg-white shadow-md rounded-lg p-4 sm:p-6 hover:shadow-lg transition-shadow duration-300">
            <h3 class="text-lg sm:text-xl font-semibold mb-2">Estimating accurate traffic time by smart simulative route predictions</h3>
            <p class="text-gray-600 mb-2 text-sm sm:text-base">I3E 2025</p>
            <p class="text-xs sm:text-sm text-gray-500 mb-3 sm:mb-4">Authors: Nadav Voloch, Neev Penkar, Guy Tordjman</p>
            <a href="/publications/SmartSimulativeRoute_I3E_2025_Final.pdf" target="_blank" 
               class="inline-flex items-center px-3 sm:px-4 py-1.5 sm:py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-300 text-xs sm:text-sm">
              <svg class="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
              Download PDF
            </a>
          </div>
        </div>
      </div>
    </section>

    <!-- Contact Section -->
    <section id="contact" class="py-12 sm:py-16 bg-gray-900 text-white">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 class="text-2xl sm:text-3xl font-bold mb-6 sm:mb-8">Contact Us</h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8">
          <div>
            <h3 class="text-lg sm:text-xl font-semibold mb-2">Email</h3>
            <p class="text-gray-300 text-sm sm:text-base">nadavv@ruppin.ac.il</p>
          </div>
          <div>
            <h3 class="text-lg sm:text-xl font-semibold mb-2">Location</h3>
            <p class="text-gray-300 text-sm sm:text-base">Ruppin Academic Center<br>Israel</p>
          </div>
          <div>
            <h3 class="text-lg sm:text-xl font-semibold mb-2">Phone</h3>
            <p class="text-gray-300 text-sm sm:text-base">+972 98-983-866</p>
          </div>
          <div>
            <h3 class="text-lg sm:text-xl font-semibold mb-2">GitHub</h3>
            <a href="https://github.com/Ruppin-SmartTransportation" target="_blank" 
               class="text-blue-400 hover:text-blue-300 transition duration-300 inline-flex items-center text-sm sm:text-base">
              <svg class="w-4 h-4 sm:w-5 sm:h-5 mr-1 sm:mr-2" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              View Our Code
            </a>
          </div>
        </div>
      </div>
    </section>

    <!-- Funding Section -->
    <section class="py-6 sm:py-8 bg-gray-800 text-white">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h3 class="text-base sm:text-lg font-semibold mb-2">Funding</h3>
        <p class="text-gray-300 text-xs sm:text-sm">
          This research is supported by the Israeli Ministry of Innovation, Science and Technology 
          (Proposal no. 0007846) through grant number 34836.
        </p>
      </div>
    </section>

    <!-- Footer -->
    <footer class="bg-gray-800 text-white py-6 sm:py-8">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <p class="text-sm sm:text-base">&copy; 2025 SmartTransportation Lab. All rights reserved.</p>
        <p class="text-xs sm:text-sm text-gray-400 mt-2">
          API Status: 
          <span :class="apiStatus === 'healthy' ? 'text-green-400' : 'text-red-400'">
            {{ apiStatus }}
          </span>
        </p>
      </div>
    </footer>
  </div>
</template>

<style scoped>
/* Fix for white border/padding on right side in portrait mode */
@media (orientation: portrait) and (max-width: 1024px) {
  .max-w-7xl {
    max-width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
  }
  
  /* Ensure all sections take full width */
  section {
    width: 100% !important;
    max-width: 100% !important;
  }
}

/* Ensure full width on mobile devices */
@media (max-width: 640px) {
  .max-w-7xl {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }
  
  /* Ensure all sections take full width on mobile */
  section {
    width: 100% !important;
    max-width: 100% !important;
  }
}

/* Ensure no horizontal overflow */
* {
  box-sizing: border-box;
}

/* Fix any potential overflow issues */
:deep(body) {
  overflow-x: hidden;
}

/* Additional fix for portrait mode */
@media (orientation: portrait) {
  .w-full {
    max-width: 100vw !important;
  }
}

</style>

<script>
import apiService from '../services/api.js'

export default {
  name: 'HomePage',
  data() {
    return {
      apiStatus: 'checking...',
      showImage: true,
      showImage2: true,
      showImage3: true,
      showImage4: true
    }
  },
  methods: {
    handleImageError(event) {
      // Hide image and show fallback
      this.showImage = false
    },
    handleImageLoad(event) {
      // Show image when it loads successfully
      this.showImage = true
    },
    handleImageError2(event) {
      // Hide image and show fallback for Guy's image
      this.showImage2 = false
    },
    handleImageLoad2(event) {
      // Show image when it loads successfully for Guy's image
      this.showImage2 = true
    },
    handleImageError3(event) {
      // Hide image and show fallback for Maor's image
      this.showImage3 = false
    },
    handleImageLoad3(event) {
      // Show image when it loads successfully for Maor's image
      this.showImage3 = true
    },
    handleImageError4(event) {
      // Hide image and show fallback for Matan's image
      this.showImage4 = false
    },
    handleImageLoad4(event) {
      // Show image when it loads successfully for Matan's image
      this.showImage4 = true
    }
  },
  async mounted() {
    // Check API health
    try {
      const health = await apiService.getHealth()
      this.apiStatus = health.status
    } catch (error) {
      this.apiStatus = 'offline'
      console.error('API health check failed:', error)
    }
  }
}
</script>
