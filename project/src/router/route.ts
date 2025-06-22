export default [
    {
        path: "/login",
        component: () => import("@/views/login/index.vue"),
        name: "Login"
    },
    {
        path: "/",
        component: () => import("@/layout/index.vue"),
        name: 'Layout',
        redirect:'/grid',
        children: [
            {
                path: "grid",
                component: () => import("@/views/grid/index.vue"),
                name: 'Grid',
            },
            {
                path: "analyze",
                component: () => import("@/views/analyze/index.vue"),
                name: 'Analyze',
            },
            {
                path: "predict",
                component: () => import("@/views/predict/index.vue"),
                name: 'Predict',
            },
            {
                path: "compare",
                component: () => import("@/views/compare/index.vue"),
                name: 'Compare',
            }
        ]
    }
]