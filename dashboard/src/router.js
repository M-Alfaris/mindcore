import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Overview',
    component: () => import('./views/Overview.vue')
  },
  {
    path: '/messages',
    name: 'Messages',
    component: () => import('./views/Messages.vue')
  },
  {
    path: '/threads',
    name: 'Threads',
    component: () => import('./views/Threads.vue')
  },
  {
    path: '/threads/:threadId',
    name: 'ThreadDetail',
    component: () => import('./views/ThreadDetail.vue')
  },
  {
    path: '/logs',
    name: 'Logs',
    component: () => import('./views/Logs.vue')
  },
  {
    path: '/config',
    name: 'Configuration',
    component: () => import('./views/Configuration.vue')
  },
  {
    path: '/models',
    name: 'Models',
    component: () => import('./views/Models.vue')
  },
  {
    path: '/performance',
    name: 'Performance',
    component: () => import('./views/Performance.vue')
  },
  {
    path: '/sessions',
    name: 'Sessions',
    component: () => import('./views/Sessions.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
