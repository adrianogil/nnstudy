/* Main function, uniforms & utils */
#ifdef GL_ES
    precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

#define PI_TWO			1.570796326794897
#define PI				3.141592653589793
#define TWO_PI			6.283185307179586

/* Coordinate and unit utils */
vec2 coord(in vec2 p) {
    p = p / u_resolution.xy;
    // correct aspect ratio
    if (u_resolution.x > u_resolution.y) {
        p.x *= u_resolution.x / u_resolution.y;
        p.x += (u_resolution.y - u_resolution.x) / u_resolution.y / 2.0;
    } else {
        p.y *= u_resolution.y / u_resolution.x;
        p.y += (u_resolution.x - u_resolution.y) / u_resolution.x / 2.0;
    }
    // centering
    p -= 0.5;
    p *= vec2(-1.0, 1.0);
    return p;
}
#define rx 1.0 / min(u_resolution.x, u_resolution.y)
#define uv gl_FragCoord.xy / u_resolution.xy
#define st coord(gl_FragCoord.xy)
#define mx coord(u_mouse)
#define mapcoord(X) (2.0 * (X))

uniform float u_bias;
float tansig(float n) {
    return (exp(n) - exp(-n)) / (exp(n) + exp(-n));
}
#define neural_function tansig

float neuron(vec2 p, vec2 w, float bias)
{
    return neural_function(
        p.x * w.x + p.y * w.y + bias
    );
}

void main() {
    vec2 weight = mapcoord(mx);
    vec2 neural_input = mapcoord(st);

    float neural_output = neuron(neural_input, weight, u_bias);

    neural_output = 0.5 + 0.5 * neural_output;

    vec3 color = vec3(
        neural_output,
        neural_output,
        neural_output
    );
    gl_FragColor = vec4(color, 1.0);
}
