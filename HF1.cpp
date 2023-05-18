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

const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1);		// transform vp from modeling space to normalized device space
	}
)";

const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";


GPUProgram gpuProgram; 

const float hami_elore = .01f;
const float hami_forog = .1f;

float hb_skalar_szorzat(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y - a.z * b.z; }
float hb_tavolsag(vec3 p, vec3 q) { return acoshf(-hb_skalar_szorzat(p, q)); }
vec3 hb_normalizalt(vec3 v) { return v * (1.0f / sqrtf(hb_skalar_szorzat(v, v))); }
vec2 hb_vetites(vec3 p) { return { p.x / (p.z + 1), p.y / (p.z + 1) }; }

vec3 hb_meroleges(vec3 p, vec3 v)
{
	return cross({ p.x, p.y, -p.z }, { v.x, v.y, -v.z });
}

std::pair<vec3, vec3> hb_pont_sebesseg(vec3 p, vec3 v, float t)
{
	return { p * coshf(t) + v * sinhf(t), p * sinhf(t) + v * coshf(t) };
}

std::pair<vec3, float> hb_irany_tavolsag(vec3 p, vec3 q)
{
	float s = hb_tavolsag(p, q);
	return { (q - p * coshf(s)) / sinhf(s) , s };
}

vec3 hb_irany_tavvolsagbol_pont(vec3 p, vec3 v, float s)
{
	v = hb_normalizalt(v);
	return p * coshf(s) + v * sinhf(s);
}

vec3 hb_vektor_elforgat(vec3 p, vec3 v, float fi)
{
	return { v * cosf(fi) + hb_meroleges(p, v) * sinf(fi) };
}

std::pair<vec3, vec3> hb_geometriai_kenyszer(vec3 p, vec3 v)
{
	vec3 korrigalt_p{ p.x, p.y, sqrtf(p.x * p.x + p.y * p.y + 1) };
	vec3 korrigalt_v{ hb_skalar_szorzat(korrigalt_p, v) * korrigalt_p + v };
	return { korrigalt_p, korrigalt_v };
}

class hbKor {
	vec3 p;
	float r;
	vec3 s;
	size_t f;
	unsigned int vao, vbo;
public:
	hbKor(vec3 poz = { 0, 0, 1 }, float sugar = 0.1f, vec3 szin = { 1, 1, 1 }, size_t felb = 64)
		: p{ poz }, r{ sugar }, s{ szin }, f { felb }
	{
		std::vector<vec2> csp{};

		float fordulasiszog = 2.0f * M_PI / (float)f;
		vec3 iranyvektor{ 1.0f, 0.0f, 0.0f };
		std::pair<vec3, vec3> korrigalt{ hb_geometriai_kenyszer(p, iranyvektor) };
		p = korrigalt.first;
		iranyvektor = korrigalt.second;

		for (size_t idx = 0; idx != f; ++idx) {
			vec3 fogatottvektor{ hb_vektor_elforgat(p, iranyvektor, idx * fordulasiszog) };
			vec3 csucspont{ hb_irany_tavvolsagbol_pont(p, fogatottvektor, r) };
			vec2 vetitettpont{ hb_vetites(csucspont) };
			csp.push_back(vetitettpont);
		}

		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * csp.size(), &csp[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	void draw()
	{
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, s.x, s.y, s.z);

		glBindVertexArray(vao);

		glDrawArrays(GL_TRIANGLE_FAN, 0, f);
	}
};

class hbCsiganyal {
	std::vector<vec2> pv;
	unsigned int vao, vbo;
public:
	void create(vec3 kezdeti)
	{
		pv.push_back(hb_vetites(kezdeti));
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
	}
	void add(vec3 uj) { pv.push_back(hb_vetites(uj)); }
	void draw()
	{
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.f, 1.f, 1.f);

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * pv.size(), &pv[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glDrawArrays(GL_LINE_STRIP, 0, pv.size());
	}
};

class UfoHami {
	float sugar;
	vec3 szin;
	vec3 hely;
	vec3 irany;
	UfoHami* masik;
	hbCsiganyal bejart;
	float szajszorzo;
public:
	bool mozgas;

	void create(vec3 hely, float sugar, vec3 szin, vec3 irany)
	{
		this->sugar = sugar;
		this->szin = szin;
		std::pair<vec3, vec3> korrigaltak{hb_geometriai_kenyszer(hely, irany)};
		this->hely = korrigaltak.first;
		this->irany = korrigaltak.second;
		masik = nullptr;
		bejart.create(hely);
		szajszorzo = 1.f;
		mozgas = false;
	}
	void set_masik(UfoHami* m) { masik = m; }
	void korrigal()
	{
		std::pair<vec3, vec3> korrigalt = hb_geometriai_kenyszer(hely, irany);
		hely = korrigalt.first;
		irany = hb_normalizalt(korrigalt.second);
	}
	void mozog(float ds)
	{
		hely = hb_irany_tavvolsagbol_pont(hely, irany, ds);
		korrigal();
		bejart.add(hely);
	}
	void forog(float fi)
	{
		irany = hb_vektor_elforgat(hely, irany, fi);
		korrigal();
	}
	void draw()
	{
		bejart.draw();

		hbKor alap{hely, sugar, szin};
		alap.draw();

		hbKor szaj{ hb_irany_tavvolsagbol_pont(hely, irany, sugar), .05f * szajszorzo, {0,0,0} };
		szaj.draw();

		vec3 szem1poz{ hb_irany_tavvolsagbol_pont(hely, hb_vektor_elforgat(hely, irany, +M_PI / 6.f), sugar) };
		vec3 szem2poz{ hb_irany_tavvolsagbol_pont(hely, hb_vektor_elforgat(hely, irany, -M_PI / 6.f), sugar) };
		hbKor szem1{ szem1poz, .05f };
		hbKor szem2{ szem2poz, .05f };
		szem1.draw();
		szem2.draw();

		std::pair<vec3, float> eltolas = hb_irany_tavolsag(this->hely, masik->hely);
		hbKor pupilla1{ hb_irany_tavvolsagbol_pont(szem1poz, eltolas.first, .04f) , .02f, {0,0,1} };
		hbKor pupilla2{ hb_irany_tavvolsagbol_pont(szem2poz, eltolas.first, .04f) , .02f, {0,0,1} };
		pupilla1.draw();
		pupilla2.draw();
	}
	void setszajszorzo(float f) { szajszorzo = f; }
};

UfoHami player_hami;
UfoHami ai_hami;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	gpuProgram.create(vertexSource, fragmentSource, "outColor");

	player_hami.create({ -0.7f, 0.f, 1.f }, .2f, { 1.f, 0.f, 0.f }, { 1.f, 0.f, 0.f });
	ai_hami.create({ 1.f, 0.f, 1.f }, .2f, { 0.f, 1.f, 0.f }, { 1.f, 0.f, .0f });

	player_hami.set_masik(&ai_hami);
	ai_hami.set_masik(&player_hami);
}

void onDisplay() {
	glClearColor(.5f, .5f, .5f, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	hbKor hatter{ {0.f, 0.f, 0.f}, 10.0f, {0.f,0.f,0.f} };
	hatter.draw();

	player_hami.draw();
	ai_hami.draw();

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key)
	{
	case 'e':
		player_hami.mozgas = true;
		break;
	case 's':
		player_hami.forog(hami_forog);
		break;
	case 'f':
		player_hami.forog(-hami_forog);
		break;
	}
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

long lasttime = 0;
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	long dt = (time - lasttime);

	player_hami.setszajszorzo(sinf(time / 250.f));
	ai_hami.setszajszorzo(sinf(time / 250.f));
	for (long i = lasttime; i < time; i += 5) {
		ai_hami.mozog(hami_elore / 2.f);
		ai_hami.forog(hami_forog / 10.f);
		if (player_hami.mozgas)
			player_hami.mozog(hami_elore * 2.f);
	}
	player_hami.mozgas = false;

	lasttime = time;
	glutPostRedisplay();
}
