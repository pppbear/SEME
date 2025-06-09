import { createRouter, createWebHashHistory } from "vue-router";
import routes from "./route"
import { useUserStore } from "@/stores/user";
import pinia from "@/stores/store";

const userStore = useUserStore(pinia)

const router = createRouter({
    history: createWebHashHistory(),
    routes: routes
})

router.beforeEach((to, from) => {
    console.log(from.name, to.name);
    
    if(to.name !== 'Login' && userStore.getToken() === "") {
        return {name: 'Login'}
    }
    if(to.name === 'Login' && userStore.getToken() !== "") {
        return false
    }
})

router.afterEach((to, from) => {
    // console.log(from.name, to.name);
    
})

export default router