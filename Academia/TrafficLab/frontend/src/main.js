import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import HomePage from './components/HomePage.vue'
import DemoPage from './components/DemoPage.vue'
import SimDemoPage from './components/SimDemoPage.vue'
import './style.css'

const routes = [
  { path: '/', component: HomePage },
  { path: '/demo', component: DemoPage },
  { path: '/sim-demo', component: SimDemoPage }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

const app = createApp(App)
app.use(router)
app.mount('#app')
