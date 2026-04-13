<template>
  <div id="app">
    <!-- Navigation - Only show on home page -->
    <nav v-if="$route.path === '/'" class="bg-blue-900 text-white shadow-lg">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center h-16">
          <!-- Logo/Brand -->
          <div class="flex items-center">
            <router-link to="/" class="text-lg sm:text-xl font-bold hover:text-blue-200">
              SmartTransportation Lab
            </router-link>
          </div>
          
          <!-- Desktop Navigation -->
          <div class="hidden md:flex items-center space-x-6 lg:space-x-8">
            <router-link to="/" class="hover:text-blue-200 text-sm lg:text-base">Home</router-link>
            <a href="/#about" class="hover:text-blue-200 text-sm lg:text-base">About</a>
            <a href="/#team" class="hover:text-blue-200 text-sm lg:text-base">Team</a>
            <a href="/#research" class="hover:text-blue-200 text-sm lg:text-base">Research</a>
            <router-link to="/sim-demo" class="hover:text-blue-200 text-sm lg:text-base">Live Demo</router-link>
            <a href="/#publications" class="hover:text-blue-200 text-sm lg:text-base">Publications</a>
            <a href="/#contact" class="hover:text-blue-200 text-sm lg:text-base">Contact</a>
          </div>
          
          <!-- Mobile Menu Button -->
          <div class="md:hidden">
            <button 
              @click="toggleMobileMenu"
              class="text-white hover:text-blue-200 focus:outline-none focus:text-blue-200"
              aria-label="Toggle mobile menu"
            >
              <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path v-if="!isMobileMenuOpen" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
                <path v-else stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
        
        <!-- Mobile Navigation Menu -->
        <div v-if="isMobileMenuOpen" class="md:hidden">
          <div class="px-2 pt-2 pb-3 space-y-1 bg-blue-800 rounded-lg mt-2">
            <router-link 
              to="/" 
              @click="closeMobileMenu"
              class="block px-3 py-2 text-base font-medium hover:text-blue-200 hover:bg-blue-700 rounded-md"
            >
              Home
            </router-link>
            <a 
              href="/#about" 
              @click="closeMobileMenu"
              class="block px-3 py-2 text-base font-medium hover:text-blue-200 hover:bg-blue-700 rounded-md"
            >
              About
            </a>
            <a 
              href="/#team" 
              @click="closeMobileMenu"
              class="block px-3 py-2 text-base font-medium hover:text-blue-200 hover:bg-blue-700 rounded-md"
            >
              Team
            </a>
            <a 
              href="/#research" 
              @click="closeMobileMenu"
              class="block px-3 py-2 text-base font-medium hover:text-blue-200 hover:bg-blue-700 rounded-md"
            >
              Research
            </a>
            <router-link 
              to="/sim-demo" 
              @click="closeMobileMenu"
              class="block px-3 py-2 text-base font-medium hover:text-blue-200 hover:bg-blue-700 rounded-md"
            >
              Live Demo
            </router-link>
            <a 
              href="/#publications" 
              @click="closeMobileMenu"
              class="block px-3 py-2 text-base font-medium hover:text-blue-200 hover:bg-blue-700 rounded-md"
            >
              Publications
            </a>
            <a 
              href="/#contact" 
              @click="closeMobileMenu"
              class="block px-3 py-2 text-base font-medium hover:text-blue-200 hover:bg-blue-700 rounded-md"
            >
              Contact
            </a>
          </div>
        </div>
      </div>
    </nav>

    <!-- Router View -->
    <router-view></router-view>
  </div>
</template>

<script>
export default {
  name: 'App',
  data() {
    return {
      isMobileMenuOpen: false
    }
  },
  methods: {
    toggleMobileMenu() {
      this.isMobileMenuOpen = !this.isMobileMenuOpen
    },
    closeMobileMenu() {
      this.isMobileMenuOpen = false
    }
  },
  mounted() {
    // Close mobile menu when clicking outside
    document.addEventListener('click', (event) => {
      if (this.isMobileMenuOpen && !event.target.closest('nav')) {
        this.closeMobileMenu()
      }
    })
    
    // Close mobile menu on route change
    this.$router.afterEach(() => {
      this.closeMobileMenu()
    })
  }
}
</script>
