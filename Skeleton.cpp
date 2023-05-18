//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const float epsilon = 0.0001f;

struct Hit {
	float t;
	vec3 position, normal;

	Hit()
		: t{ -1 }, position{ }, normal{ }
	{ }

	explicit Hit(float _t, const vec3& _position, const vec3& _normal)
		: t{ _t }, position{ _position }, normal{ normalize(_normal) }
	{ }
};

struct Ray {
	vec3 start, dir;

	Ray(vec3 _start, vec3 _dir)
		: start{ _start }, dir{ normalize(_dir) }
	{ }
};

class Intersectable {
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

class  TriPlane_t : public Intersectable {
private:
	const vec3 m_r1, m_r2, m_r3, m_normal;
public:
	TriPlane_t(const vec3& r1, const vec3& r2, const vec3& r3)
		: m_r1{ r1 }, m_r2{ r2 }, m_r3{ r3 }, m_normal{ normalize(cross(r2 - r1, r3 - r1)) }
	{ }
	
	virtual Hit intersect(const Ray& ray) override {
		const float t = dot((m_r1 - ray.start), m_normal) / dot(ray.dir, m_normal);
		if (t < 0) return Hit{};
		const vec3 p = ray.start + ray.dir * t;
		if (dot(cross(m_r2 - m_r1, p - m_r1), m_normal) > 0 &&
			dot(cross(m_r3 - m_r2, p - m_r2), m_normal) > 0 &&
			dot(cross(m_r1 - m_r3, p - m_r3), m_normal) > 0) {
			return Hit{ t, p, m_normal};
		}
		return Hit{};
	};
};

class NFacePlane_t : public Intersectable {
private:
	std::vector<TriPlane_t> m_triPlanes;
public:
	NFacePlane_t(const std::initializer_list<vec3>& vertices)
		: m_triPlanes{ }
	{
		for (size_t idx = 1; idx < vertices.size() - 1; idx++) {
			m_triPlanes.emplace_back(*(vertices.begin()), *(vertices.begin() + idx), *(vertices.begin() + idx + 1));
		}
	}

	virtual Hit intersect(const Ray& ray) override {
		Hit bestHit;
		for (auto& triPlane : m_triPlanes) {
			Hit hit = triPlane.intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		return bestHit;
	}
};

class HollowRoom_t : public Intersectable {
private:
	std::vector<NFacePlane_t> m_faces;
	const vec3 a{ 0, 0, 0 }, b{ 4,0,0 }, c{ 4,0,4 }, d{ 0,0,4 }, e{ 0,4,0 }, f{ 4,4,0 }, g{ 4,4,4 }, h{ 0,4,4 };
public:
	HollowRoom_t()
		: m_faces{ }
	{
		m_faces.push_back({ {a, b, c, d} });
		m_faces.push_back({ {a, b, f, e} });
		m_faces.push_back({ {b, c, g, f} });
		m_faces.push_back({ {c, d, h, g} });
		m_faces.push_back({ {d, a, e, h} });
		m_faces.push_back({ {e, f, g, h} });
	}

	virtual Hit intersect(const Ray& ray) override {
		Hit secondHit;
		for (auto& face : m_faces) {
			Hit hit = face.intersect(ray);
			if (hit.t > 0 && (secondHit.t < 0 || hit.t > secondHit.t)) secondHit = hit;
		}
		return secondHit;
	}
};

class ConeLight_t : public Intersectable {
private:
	vec3 m_position, m_direction;
	const vec3 m_color;
	const float m_alpha, m_length;
public:
	ConeLight_t(vec3 position, vec3 direction, vec3 color, float alpha = .5f, float length = .25f)
		: m_position{ position }, m_direction{ normalize(direction) }, m_color{ color }, m_alpha{ alpha }, m_length{ length }
	{}

	void reset(vec3 pos, vec3 dir) { m_position = pos; m_direction = dir; }

	virtual Hit intersect(const Ray &ray) override {
		const float a = dot(ray.dir, m_direction) * dot(ray.dir, m_direction) - cosf(m_alpha) * cosf(m_alpha);
		const float b = 2.f * (dot(ray.dir, m_direction) * dot(ray.start - m_position, m_direction) - dot(ray.dir, ray.start - m_position) * cosf(m_alpha) * cosf(m_alpha));
		const float c = dot(ray.start - m_position, m_direction) * dot(ray.start - m_position, m_direction) - dot(ray.start - m_position, ray.start - m_position) * cosf(m_alpha) * cosf(m_alpha);
		float discr = b * b - 4 * a * c;
		if (discr < 0) return Hit{};
		else discr = sqrtf(discr);
		float t1 = (-b + discr) / 2.f / a, t2 = (-b - discr) / 2.f / a;
		if (t1 <= 0) return Hit{ };
		float t = (t2 < 0) ? t2 : t1;
		vec3 p = ray.start + ray.dir * t;
		if (dot((p - m_position), m_direction) < 0 || dot((p - m_position), m_direction) > m_length) {
			t = (t2 < 0) ? t1 : t2;
			p = ray.start + ray.dir * t;
			if (dot((p - m_position), m_direction) < 0 || dot((p - m_position), m_direction) > m_length)
				return Hit{};
		}
		Hit hit;
		hit.t = t;
		hit.position = p;
		hit.normal = 2.f * dot(hit.position - m_position, m_direction) * m_direction - 2.f * (hit.position - m_position) * cosf(m_alpha) * cosf(m_alpha);
		return hit;
	}

	vec3 getPos() const { return m_position; }
	vec3 getDir() const { return m_direction; }
	vec3 getCol() const { return m_color; }

	bool pointIsInCOne(vec3 point) const {
		if (dot(normalize(point - m_position), m_direction) < cosf(m_alpha) - epsilon*10) return false;
		return true;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (windowWidth - X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};


struct Scene {
	std::vector<Intersectable*> objects;
	std::vector<ConeLight_t*> conelights;
	Camera camera;
	vec3 La;

	void build() {
		vec3 eye = vec3(-4.3f, 2.5f, -3.7f), vup = vec3(0.f, 1.f, 0.f), lookat = vec3(2.f, 2.f, 2.f);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(.8f, .8f, .8f);
		
		objects.push_back(new HollowRoom_t{  });

		const vec3 i{.5f,0,3.5f};
		const vec3 j{.5f,0,1.5f};
		const vec3 k{2.23f,0,2.5f};
		const vec3 l{1.08f,1.63,2.5f};
		objects.push_back(new TriPlane_t{ i,j,k });
		objects.push_back(new TriPlane_t{ i,j,l });
		objects.push_back(new TriPlane_t{ j,k,l });
		objects.push_back(new TriPlane_t{ k,i,l });

		// ikozaeder pontos adatokat a kiadott oldalon talalt https://people.sc.fsu.edu/~jburkardt/data/obj/icosahedron.obj url-rol szedtem, majd ezekre offsetet allitottam
		const float icox = 2.f, icoy = 2.f, icoz = 1.f;
		const vec3  m{ 0 + icox, -0.525731 + icoy, 0.850651 + icoz };
		const vec3 n{ 0.850651 + icox, 0 + icoy, 0.525731 + icoz };
		const vec3 o{ 0.850651 + icox, 0 + icoy, -0.525731 + icoz };
		const vec3 p{ -0.850651 + icox, 0 + icoy, -0.525731 + icoz };
		const vec3 q{ -0.850651 + icox, 0 + icoy, 0.525731 + icoz };
		const vec3 r{ -0.525731 + icox, 0.850651 + icoy, 0 + icoz };
		const vec3 s{ 0.525731 + icox, 0.850651 + icoy, 0 + icoz };
		const vec3 t{ 0.525731 + icox, -0.850651 + icoy, 0 + icoz };
		const vec3 v{ -0.525731 + icox, -0.850651 + icoy, 0 + icoz };
		const vec3 w{ 0 + icox, -0.525731 + icoy, -0.850651 + icoz };
		const vec3 x{ 0 + icox, 0.525731 + icoy, -0.850651 + icoz };
		const vec3 y{ 0 + icox, 0.525731 + icoy, 0.850651 + icoz };
		objects.push_back(new TriPlane_t{ n,o,s });
		objects.push_back(new TriPlane_t{ n,t,o });
		objects.push_back(new TriPlane_t{ p,q,r });
		objects.push_back(new TriPlane_t{ q,p,v });
		objects.push_back(new TriPlane_t{ s,r,y });
		objects.push_back(new TriPlane_t{ r,s,x });
		objects.push_back(new TriPlane_t{ w,x,o });
		objects.push_back(new TriPlane_t{ x,w,p });
		objects.push_back(new TriPlane_t{ t,v,w });
		objects.push_back(new TriPlane_t{ v,t,m });
		objects.push_back(new TriPlane_t{ y,m,n });
		objects.push_back(new TriPlane_t{ m,y,q });
		objects.push_back(new TriPlane_t{ s,o,x });
		objects.push_back(new TriPlane_t{ n,s,y });
		objects.push_back(new TriPlane_t{ p,r,x });
		objects.push_back(new TriPlane_t{ r,q,y });
		objects.push_back(new TriPlane_t{ o,t,w });
		objects.push_back(new TriPlane_t{ t,n,m });
		objects.push_back(new TriPlane_t{ p,w,v });
		objects.push_back(new TriPlane_t{ q,v,m });

		ConeLight_t* coner = new ConeLight_t{ {0,4,4}, {1,-1,-1}, {1,0,0} };
		ConeLight_t* coneg = new ConeLight_t{ {3,4,4}, {-.5f,-1,-1}, {0,1,0} };
		ConeLight_t* coneb = new ConeLight_t{ {4,4,0}, {-1,-1,1}, {0,0,1} };
		conelights.push_back(coner); objects.push_back(coner);
		conelights.push_back(coneg); objects.push_back(coneg);
		conelights.push_back(coneb); objects.push_back(coneb);
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return { };

		float cosThetaOut = dot(normalize(hit.normal), normalize(ray.dir));
		vec3 outRadiance = .2f * (1.f + fabs(cosThetaOut)) * La;

		for (ConeLight_t* conelight : conelights) {
			if (!conelight->pointIsInCOne(hit.position)) continue;

			const vec3 lightpos = conelight->getPos() + conelight->getDir() * epsilon * 100;

			Hit lighthit = firstIntersect(Ray{ lightpos, hit.position - lightpos });
			if (length(lighthit.position - hit.position) > epsilon * 100 && conelight->pointIsInCOne(lighthit.position)) lighthit.t = -1;

			if (lighthit.t > 0) {
				outRadiance = outRadiance + conelight->getCol() / (sqrtf( lighthit.t));
			}
		}
		return outRadiance;
	}

	Ray getRay(int X, int Y) {
		return camera.getRay(X, Y);
	}
};

GPUProgram gpuProgram;
Scene scene;

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;
	out vec4 fragmentColor;

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);

	delete fullScreenTexturedQuad;
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

ConeLight_t* getNearestConeLight(vec3 point) {
	float d = 0;
	ConeLight_t* best = nullptr;

	for (auto conelight : scene.conelights) {
		const float newd = length(conelight->getPos() - point);
		if (best == nullptr || newd < d) {
			best = conelight;
			d = newd;
		}
	}

	return best;
}

void ReplaceConeLight(int x, int y) {
	Ray ray = scene.getRay(x, y);
	Hit hit = scene.firstIntersect(ray);
	if (hit.t > 0) {
		ConeLight_t* mod = getNearestConeLight(hit.position);
		mod->reset(hit.position, hit.normal);
	}
}

void onMouse(int button, int state, int pX, int pY) {
	if (state == GLUT_DOWN)
		ReplaceConeLight(pX, windowHeight - pY);
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
}