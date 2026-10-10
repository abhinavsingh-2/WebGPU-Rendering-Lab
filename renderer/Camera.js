import { vec3, mat4 } from 'gl-matrix';

export default class Camera {
    constructor(canvas) {
        this.canvas = canvas;
        if (!canvas) {
            // Provide defaults for headless or deferred initialization
            this.eye = vec3.fromValues(4.5, 0.5, 0.5);
            this.target = vec3.fromValues(0, 0, 0);
            this.up = vec3.fromValues(0, 1, 0);
            this.viewMatrix = mat4.create();
            this.projectionMatrix = mat4.create();
            return;
        }

        // ===============================
        // CAMERA STATE
        // ===============================
        this.eye = vec3.fromValues(4.5, 0.5, 0.5);
        // 🔴 FIX: start far away
        this.target = vec3.fromValues(0, 0, 0);
        this.up = vec3.fromValues(0, 1, 0);

        this.viewMatrix = mat4.create();
        this.projectionMatrix = mat4.create();

        const aspect = canvas.width / canvas.height;
        mat4.perspectiveZO(
            this.projectionMatrix,
            (2 * Math.PI) / 5,
            aspect,
            0.1, // Near plane should be smaller for close ups
            1000000.0 // Far plane
        );

        // ===============================
        // ORBIT CONTROLS
        // ===============================
        this.radius = vec3.distance(this.eye, this.target);
        this.theta = Math.atan2(this.eye[0], this.eye[2]);
        this.phi = Math.acos(this.eye[1] / this.radius);

        let isMouseDown = false;
        let lastX = 0;
        let lastY = 0;
        let button = 0;

        this.updateOrbit = () => {
            this.radius = vec3.distance(this.eye, this.target);
            this.theta = Math.atan2(this.eye[0] - this.target[0], this.eye[2] - this.target[2]);
            this.phi = Math.acos((this.eye[1] - this.target[1]) / (this.radius || 1));
        };

        const updateEye = () => {
            this.phi = Math.max(0.01, Math.min(Math.PI - 0.01, this.phi));

            this.eye[0] = this.target[0] + this.radius * Math.sin(this.phi) * Math.sin(this.theta);
            this.eye[1] = this.target[1] + this.radius * Math.cos(this.phi);
            this.eye[2] = this.target[2] + this.radius * Math.sin(this.phi) * Math.cos(this.theta);

            mat4.lookAt(this.viewMatrix, this.eye, this.target, this.up);
        };

        updateEye();

        // ===============================
        // MOUSE EVENTS
        // ===============================
        canvas.oncontextmenu = e => e.preventDefault();

        canvas.onmousedown = e => {
            isMouseDown = true;
            button = e.button;
            lastX = e.clientX;
            lastY = e.clientY;
            this.updateOrbit();
        };

        window.onmouseup = () => {
            isMouseDown = false;
        };

        window.onmousemove = e => {
            if (!isMouseDown) return;

            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;

            if (button === 0) {
                // LEFT: orbit
                this.theta -= dx * 0.005;
                this.phi -= dy * 0.005;
            } else if (button === 2) {
                // RIGHT: pan
                const panSpeed = this.radius * 0.001;
                const right = vec3.create();
                const forward = vec3.create();

                vec3.sub(forward, this.target, this.eye);
                vec3.normalize(forward, forward);
                vec3.cross(right, forward, this.up);
                vec3.normalize(right, right);

                vec3.scaleAndAdd(this.eye, this.eye, right, -dx * panSpeed);
                vec3.scaleAndAdd(this.eye, this.eye, this.up, dy * panSpeed);
                vec3.scaleAndAdd(this.target, this.target, right, -dx * panSpeed);
                vec3.scaleAndAdd(this.target, this.target, this.up, dy * panSpeed);
            }

            updateEye();
            lastX = e.clientX;
            lastY = e.clientY;
        };

        window.onwheel = e => {
            this.radius += e.deltaY * (this.radius * 0.001);
            this.radius = Math.max(0.1, this.radius);
            updateEye();
        };
    }

    updateView() {
        mat4.lookAt(this.viewMatrix, this.eye, this.target, this.up);
    }

    updateProjection() {
        const aspect = this.canvas.width / this.canvas.height;
        mat4.perspectiveZO(
            this.projectionMatrix,
            (2 * Math.PI) / 5,
            aspect,
            0.1,
            1000000.0
        );
    }

    update() {
        // no-op (kept for compatibility)
    }

    getPosition() {
        return this.eye;
    }
}

