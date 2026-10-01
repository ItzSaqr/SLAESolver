#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <stdexcept>
#define M_PI 3.14159265358979323846

using namespace std;

struct Matrix {
    vector<int> ig;
    vector<int> jg;
    vector<double> ggl;
    vector<double> ggu;
    vector<double> diag;
};

struct ProfileMatrix {
    vector<int> ia;       // size n+1
    vector<double> al;    // lower profile
    vector<double> au;    // upper profile
    vector<double> diag;  // diagonal
};

int first_col_in_row(const Matrix& A, int i) {
    if (A.ig[i] == A.ig[i + 1]) return i;
    return A.jg[A.ig[i]];
}

void convert_to_profile(const Matrix& A, ProfileMatrix& P, int n) {
    P.ia.assign(n + 1, 0);
    P.diag = A.diag;

    // строим ia
    for (int i = 0; i < n; ++i) {
        int j0 = first_col_in_row(A, i);
        int width = i - j0;
        P.ia[i + 1] = P.ia[i] + width;
    }

    int m = P.ia[n];
    P.al.assign(m, 0.0);
    P.au.assign(m, 0.0);

    // переносим значения
    for (int i = 0; i < n; ++i) {
        int j0 = first_col_in_row(A, i);
        int width = i - j0;
        int start = P.ia[i];

        // сначала профиль заполнен нулями
        for (int t = 0; t < width; ++t) {
            P.al[start + t] = 0.0;
            P.au[start + t] = 0.0;
        }

        // переносим реальные нижние элементы строки i
        for (int k = A.ig[i]; k < A.ig[i + 1]; ++k) {
            int j = A.jg[k];
            int pos = start + (j - j0);
            P.al[pos] = A.ggl[k];
            P.au[pos] = A.ggu[k];
        }
    }
}

int profile_first_col(const ProfileMatrix& P, int i) {
    return i - (P.ia[i + 1] - P.ia[i]);
}

double getL(const ProfileMatrix& P, int i, int j) {
    if (j >= i) return 0.0;
    int j0 = profile_first_col(P, i);
    if (j < j0) return 0.0;
    return P.al[P.ia[i] + (j - j0)];
}

double getU(const ProfileMatrix& P, int i, int j) {
    if (j >= i) return 0.0;
    int j0 = profile_first_col(P, i);
    if (j < j0) return 0.0;
    return P.au[P.ia[i] + (j - j0)];
}

void setL(ProfileMatrix& P, int i, int j, double val) {
    int j0 = profile_first_col(P, i);
    P.al[P.ia[i] + (j - j0)] = val;
}

void setU(ProfileMatrix& P, int i, int j, double val) {
    int j0 = profile_first_col(P, i);
    P.au[P.ia[i] + (j - j0)] = val;
}

struct Grid3D {
    int nx = 0, ny = 0, nz = 0;     // число узлов по осям
    double x0 = 0.0, x1 = 1.0;
    double y0 = 0.0, y1 = 1.0;
    double z0 = 0.0, z1 = 1.0;

    double hx = 0.0, hy = 0.0, hz = 0.0;

    int nodes_count = 0;            // обычных пространственных узлов
    int total_unknowns = 0;         // 2 * nodes_count для us и uc
};

struct HarmonicData {
    double lambda = 1.0; // коэффициент при grad u
    double sigma = 1.0;  // коэффициент при du/dt
    double c = 1.0;      // коэффициент при d2u/dt2
    double omega = 1.0;
};

struct SLAE {
    Matrix A;
    Matrix M;
    vector<double> b;
    vector<double> x;
    int n = 0;
    int maxiter = 0;
    double epsilon = 0.0;

    Grid3D grid;
    HarmonicData phys;
};

double us_exact(double x, double y, double z, int testId) {
    switch (testId) {
    case 1: return x + y + z;
    case 2: return sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
    default: return 0.0;
    }
}

double uc_exact(double x, double y, double z, int testId) {
    switch (testId) {
    case 1: return 0.0;
    case 2: return 0.0;
    default: return 0.0;
    }
}

double us_boundary(double x, double y, double z, int testId) {
    return us_exact(x, y, z, testId);
}

double uc_boundary(double x, double y, double z, int testId) {
    return uc_exact(x, y, z, testId);
}

double fs_rhs(double x, double y, double z, const HarmonicData& p, int testId) {
    switch (testId) {
    case 1:
        return -(p.omega * p.omega) * p.c * (x + y + z);
    default:
        return 0.0;
    }
}

double fc_rhs(double x, double y, double z, const HarmonicData& p, int testId) {
    switch (testId) {
    case 1:
        return p.omega * p.sigma * (x + y + z);
    default:
        return 0.0;
    }
}

template<typename T>
void read_array(const string& fname, vector<T>& arr, int n) {
    ifstream f(fname);
    if (!f) {
        throw runtime_error("Cannot open file: " + fname);
    }

    arr.resize(n);
    for (int i = 0; i < n; ++i) {
        if (!(f >> arr[i])) {
            throw runtime_error("Error reading file: " + fname);
        }
    }
}

void mult_Ax(const Matrix& A, const vector<double>& x, vector<double>& res, int n) {
    res.assign(n, 0.0);

    for (int i = 0; i < n; ++i) {
        res[i] = A.diag[i] * x[i];
    }

    for (int i = 0; i < n; ++i) {
        int row_start = A.ig[i];
        int row_end = A.ig[i + 1];

        for (int k = row_start; k < row_end; ++k) {
            int j = A.jg[k]; // j < i
            double xl = x[j];
            double xi = x[i];

            res[i] += A.ggl[k] * xl;
            res[j] += A.ggu[k] * xi;
        }
    }
}

double scalar(const vector<double>& a, const vector<double>& b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

void residual(const Matrix& A, const vector<double>& x, const vector<double>& b, vector<double>& r, int n) {
    mult_Ax(A, x, r, n);
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - r[i];
    }
}

// out = alpha * a + beta * b
void vec_diff(const vector<double>& a, double alpha,
    const vector<double>& b, double beta,
    vector<double>& out, int n) {
    out.resize(n);
    for (int i = 0; i < n; ++i) {
        out[i] = alpha * a[i] + beta * b[i];
    }
}

void read_system(SLAE& sys) {
    ifstream fk("kuslau.txt");
    if (!fk) {
        throw runtime_error("Cannot open file: kuslau.txt");
    }

    fk >> sys.n >> sys.maxiter >> sys.epsilon;
    fk.close();

    int n = sys.n;

    read_array("ig.txt", sys.A.ig, n + 1);

    for (int i = 0; i <= n; ++i) {
        sys.A.ig[i]--;
    }

    int m = sys.A.ig[n];

    read_array("jg.txt", sys.A.jg, m);
    read_array("ggl.txt", sys.A.ggl, m);
    read_array("ggu.txt", sys.A.ggu, m);
    read_array("di.txt", sys.A.diag, n);
    read_array("pr.txt", sys.b, n);

    for (int i = 0; i < m; ++i) {
        sys.A.jg[i]--;
    }

    sys.x.resize(n);

    ifstream fx("x0.txt");
    if (fx) {
        fx.close();
        read_array("x0.txt", sys.x, n);
    }
    else {
        fill(sys.x.begin(), sys.x.end(), 0.0);
    }
}

void copy_matrix(Matrix& dst, const Matrix& src, int n) {
    int m = src.ig[n];

    dst.ig = src.ig;
    dst.jg = src.jg;

    dst.ggl.resize(m);
    dst.ggu.resize(m);
    dst.diag.resize(n);

    for (int i = 0; i < n; ++i) {
        dst.diag[i] = src.diag[i];
    }

    for (int i = 0; i < m; ++i) {
        dst.ggl[i] = src.ggl[i];
        dst.ggu[i] = src.ggu[i];
    }
}

void LU(Matrix& A, int n) {
    vector<int>& ig = A.ig;
    vector<int>& jg = A.jg;
    vector<double>& L = A.ggl;
    vector<double>& U = A.ggu;
    vector<double>& D = A.diag;

    for (int i = 0; i < n; ++i) {
        double sumDi = 0.0;

        int i0 = ig[i];
        int i1 = ig[i + 1];

        for (int k = i0; k < i1; ++k) {
            double sumL = 0.0;
            double sumU = 0.0;

            int j = jg[k];

            int j0 = ig[j];
            int j1 = ig[j + 1];

            int ik = i0;
            int kj = j0;

            while (ik < i1 && kj < j1) {
                int col_i = jg[ik];
                int col_j = jg[kj];

                if (col_i < col_j) {
                    ++ik;
                }
                else if (col_j < col_i) {
                    ++kj;
                }
                else {
                    sumL += L[ik] * U[kj];
                    sumU += L[kj] * U[ik];
                    ++ik;
                    ++kj;
                }
            }

            L[k] = (L[k] - sumL) / D[j];
            U[k] = (U[k] - sumU);

            sumDi += L[k] * U[k];
        }

        D[i] = D[i] - sumDi;
    }
}

// L * x = b  (x меняется на месте, на входе x=b)
void solveL_inplace(const Matrix& M, vector<double>& x, int n) {
    for (int i = 0; i < n; ++i) {
        double s = x[i];
        for (int k = M.ig[i]; k < M.ig[i + 1]; ++k) {
            int j = M.jg[k];
            s -= M.ggl[k] * x[j];
        }
        x[i] = s; // diag(L)=1
    }
}

// U * x = b
void solveU_inplace(const Matrix& M, vector<double>& x, int n) {
    for (int i = n - 1; i >= 0; --i) {
        x[i] /= M.diag[i];
        double xi = x[i];
        for (int k = M.ig[i]; k < M.ig[i + 1]; ++k) {
            int j = M.jg[k];
            x[j] -= M.ggu[k] * xi;
        }
    }
}

void mult_U(const Matrix& M, const vector<double>& x, vector<double>& y, int n) {
    y.assign(n, 0.0);

    for (int i = 0; i < n; ++i) {
        y[i] += M.diag[i] * x[i];
        for (int k = M.ig[i]; k < M.ig[i + 1]; ++k) {
            int j = M.jg[k];
            y[j] += M.ggu[k] * x[i];
        }
    }
}

void apply_Minv_LU(const SLAE& sys, vector<double>& v, vector<double>& out) {
    int n = sys.n;
    vector<double> tmp(n);
    for (int i = 0; i < n; ++i) tmp[i] = v[i];

    // M = L U, M^{-1} = U^{-1} L^{-1}
    solveL_inplace(sys.M, tmp, n);
    solveU_inplace(sys.M, tmp, n);

    for (int i = 0; i < n; ++i) out[i] = tmp[i];
}

void BiCGSTAB_LU(SLAE& sys) {
    int n = sys.n;

    vector<double> r(n);
    vector<double> r_tilde(n);
    vector<double> p(n);
    vector<double> p_new(n);
    vector<double> temp(n);
    vector<double> y(n);
    vector<double> nu(n);
    vector<double> h(n);
    vector<double> s(n);
    vector<double> z(n);
    vector<double> t(n);
    vector<double> Ks(n);
    vector<double> Kt(n);

    residual(sys.A, sys.x, sys.b, r, n);

    r_tilde = r;
    double ro = scalar(r, r_tilde, n);
    p = r;

    for (int i = 0; i < sys.maxiter; i++) {
        temp = p;
        solveL_inplace(sys.M, temp, n);
        y = temp;
        solveU_inplace(sys.M, y, n);

        mult_Ax(sys.A, y, nu, n);

        double denom_alpha = scalar(r_tilde, nu, n);
        if (fabs(denom_alpha) < 1e-30) {
            printf("Emergency exit: division by zero\n"); 
            break;
        }
        double alpha = ro / denom_alpha;

        vec_diff(sys.x, 1, y, alpha, h, n);

        vec_diff(r, 1, nu, -alpha, s, n);

        // early exit
        double normb = sqrt(scalar(sys.b, sys.b, n));
        if (normb < 1e-30) normb = 1.0;
        double s_rel = sqrt(scalar(s, s, n)) / normb;
        if (s_rel < sys.epsilon) {
            printf("Conv early on iter %d, relative res=%e\n", i + 1, s_rel);
            sys.x = h;
            break;
        }

        temp = s;
        solveL_inplace(sys.M, temp, n);
        z = temp;
        solveU_inplace(sys.M, z, n);

        mult_Ax(sys.A, z, t, n);

        Ks = s;
        solveL_inplace(sys.M, Ks, n);
        Kt = t;
        solveL_inplace(sys.M, Kt, n);
        double denom_omega = scalar(Kt, Kt, n);
        if (fabs(denom_omega) < 1e-30) {
            printf("Emergency exit: division by zero\n");
            break;
        }
        double omega = scalar(Kt, Ks, n) / denom_omega;
        if (fabs(omega) < 1e-30) {
            printf("Emergency exit: division by zero\n");
            break;
        }

        vec_diff(h, 1, z, omega, sys.x, n);

        vec_diff(s, 1, t, -omega, r, n);

        residual(sys.A, sys.x, sys.b, temp, n);
        double res = sqrt(scalar(temp, temp, n)) / sqrt(scalar(sys.b, sys.b, n));
        if (res < sys.epsilon) {
            printf("Conv early on iter %d, relative res=%e\n", i + 1, res);
            break;
        }

        double ro_new = scalar(r_tilde, r, n);
        if (fabs(ro) < 1e-30) {
            printf("Emergency exit: division by zero\n");
            break;
        }
        double beta = (ro_new / ro) * (alpha / omega);
        ro = ro_new;

        vec_diff(p, 1, nu, -omega, temp, n);
        vec_diff(r, 1, temp, beta, p_new, n);
        p = p_new;

        printf("Iter %d, res=%e\n", i + 1, res);
    }
}

void BiCGSTAB(SLAE& sys) {
    int n = sys.n;

    vector<double> r(n);
    vector<double> r_tilde(n);
    vector<double> p(n);
    vector<double> p_new(n);
    vector<double> temp(n);
    vector<double> nu(n);
    vector<double> h(n);
    vector<double> s(n);
    vector<double> t(n);

    residual(sys.A, sys.x, sys.b, r, n);

    r_tilde = r;
    double ro = scalar(r, r_tilde, n);
    p = r;

    for (int i = 0; i < sys.maxiter; i++) {
        mult_Ax(sys.A, p, nu, n);

        double denom_alpha = scalar(r_tilde, nu, n);
        if (fabs(denom_alpha) < 1e-30){
            printf("Emergency exit: division by zero\n");
            break;
        }
        double alpha = ro / denom_alpha;

        vec_diff(sys.x, 1, p, alpha, h, n);

        vec_diff(r, 1, nu, -alpha, s, n);

        // early exit
        double normb = sqrt(scalar(sys.b, sys.b, n));
        double s_rel = sqrt(scalar(s, s, n)) / normb;
        if (s_rel < sys.epsilon) {
            printf("Conv early on iter %d, relative res=%e\n", i + 1, s_rel);
            sys.x = h;
            break;
        }

        mult_Ax(sys.A, s, t, n);

        double denom_omega = scalar(t, t, n);
        if (fabs(denom_omega) < 1e-30){
            printf("Emergency exit: division by zero\n");
            break;
        }
        double omega = scalar(t, s, n) / denom_omega;
        if (fabs(omega) < 1e-30){
            printf("Emergency exit: division by zero\n");
            break;
        }

        vec_diff(h, 1, s, omega, sys.x, n);

        vec_diff(s, 1, t, -omega, r, n);

        residual(sys.A, sys.x, sys.b, temp, n);
        double res = sqrt(scalar(temp, temp, n)) / sqrt(scalar(sys.b, sys.b, n));
        if (res < sys.epsilon) {
            printf("Conv early on iter %d, relative res=%e\n", i + 1, res);
            break;
        }

        double ro_new = scalar(r_tilde, r, n);
        if (fabs(ro) < 1e-30){
            printf("Emergency exit: division by zero\n");
            break;
        }
        double beta = (ro_new / ro) * (alpha / omega);
        ro = ro_new;

        vec_diff(p, 1, nu, -omega, temp, n);
        vec_diff(r, 1, temp, beta, p_new, n);
        p = p_new;

        printf("Iter %d, res=%e\n", i + 1, res);
    }
}

void LOC_LU(SLAE& sys) {
    int n = sys.n;

    vector<double> r_tilde(n);
    vector<double> z(n);
    vector<double> p(n);
    vector<double> tmp(n);
    vector<double> Ar_til(n);
    vector<double> r_orig(n);

    residual(sys.A, sys.x, sys.b, r_orig, n);
    apply_Minv_LU(sys, r_orig, r_tilde);

    for (int i = 0; i < n; ++i)
        z[i] = r_tilde[i];

    mult_Ax(sys.A, z, tmp, n);
    apply_Minv_LU(sys, tmp, p);

    double r0_norm2 = scalar(r_orig, r_orig, n);
    double r0_norm = std::sqrt(r0_norm2);

    for (int k = 0; k < sys.maxiter; ++k) {
        double pp = scalar(p, p, n);
        if (fabs(pp) < 1e-30) {
            printf("Emergency exit: vector norm is zero\n");
            break;
        }

        double pr = scalar(p, r_tilde, n);
        double alpha = pr / pp;

        vec_diff(sys.x, 1.0, z, alpha, sys.x, n);
        vec_diff(r_tilde, 1.0, p, -alpha, r_tilde, n);

        residual(sys.A, sys.x, sys.b, r_orig, n);
        double r_norm2 = scalar(r_orig, r_orig, n);
        double rel = std::sqrt(r_norm2) / r0_norm;
        printf("Iter %d, res=%e\n", k + 1, rel);
        if (rel < sys.epsilon) break;

        mult_Ax(sys.A, r_tilde, tmp, n);
        apply_Minv_LU(sys, tmp, Ar_til);

        double beta = -scalar(p, Ar_til, n) / pp;

        vec_diff(r_tilde, 1.0, z, beta, z, n);
        vec_diff(Ar_til, 1.0, p, beta, p, n);
    }
}

void LU_profile(ProfileMatrix& P, int n) {
    for (int i = 0; i < n; ++i) {
        int j0_i = i - (P.ia[i + 1] - P.ia[i]);
        long long row_i = P.ia[i];

        for (int j = j0_i; j < i; ++j) {
            int j0_j = j - (P.ia[j + 1] - P.ia[j]);
            long long row_j = P.ia[j];

            int k0 = max(j0_i, j0_j);

            double sumL = 0.0;
            double sumU = 0.0;

            for (int k = k0; k < j; ++k) {
                double Lik = P.al[row_i + (k - j0_i)];
                double Ujk = P.au[row_j + (k - j0_j)];
                double Ljk = P.al[row_j + (k - j0_j)];
                double Uik = P.au[row_i + (k - j0_i)];

                sumL += Lik * Ujk;
                sumU += Ljk * Uik;
            }

            long long pos_ij = row_i + (j - j0_i);

            P.al[pos_ij] = (P.al[pos_ij] - sumL) / P.diag[j];
            P.au[pos_ij] = (P.au[pos_ij] - sumU);
        }

        double sumD = 0.0;
        for (int k = j0_i; k < i; ++k) {
            long long pos_ik = row_i + (k - j0_i);
            sumD += P.al[pos_ik] * P.au[pos_ik];
        }

        P.diag[i] -= sumD;
    }
}

void solveL_profile(const ProfileMatrix& P, vector<double>& x, int n) {
    for (int i = 0; i < n; ++i) {
        int j0 = i - (P.ia[i + 1] - P.ia[i]);
        long long row_i = P.ia[i];

        double s = x[i];
        for (int j = j0; j < i; ++j) {
            s -= P.al[row_i + (j - j0)] * x[j];
        }
        x[i] = s;
    }
}

void solveU_profile(const ProfileMatrix& P, vector<double>& x, int n) {
    for (int i = n - 1; i >= 0; --i) {
        int j0 = i - (P.ia[i + 1] - P.ia[i]);
        long long row_i = P.ia[i];

        x[i] /= P.diag[i];
        double xi = x[i];

        for (int j = j0; j < i; ++j) {
            x[j] -= P.au[row_i + (j - j0)] * xi;
        }
    }
}

void solveLU_direct(SLAE& sys) {
    ProfileMatrix P;
    convert_to_profile(sys.A, P, sys.n);

    LU_profile(P, sys.n);

    sys.x = sys.b;
    solveL_profile(P, sys.x, sys.n);
    solveU_profile(P, sys.x, sys.n);
}

inline int node_id(int i, int j, int k, const Grid3D& g) {
    return i + j * g.nx + k * g.nx * g.ny;
}

inline int id_s(int node, int N) {
    return node;
}

inline int id_c(int node, int N) {
    return N + node;
}

double phi_hex8(int a, double xi, double eta, double zeta) {
    static const int sx[8] = { -1, 1, 1,-1,-1, 1, 1,-1 };
    static const int sy[8] = { -1,-1, 1, 1,-1,-1, 1, 1 };
    static const int sz[8] = { -1,-1,-1,-1, 1, 1, 1, 1 };

    return 0.125 * (1.0 + sx[a] * xi)
        * (1.0 + sy[a] * eta)
        * (1.0 + sz[a] * zeta);
}

double dphi_dxi_hex8(int a, double xi, double eta, double zeta) {
    static const int sx[8] = { -1, 1, 1,-1,-1, 1, 1,-1 };
    static const int sy[8] = { -1,-1, 1, 1,-1,-1, 1, 1 };
    static const int sz[8] = { -1,-1,-1,-1, 1, 1, 1, 1 };

    return 0.125 * sx[a]
        * (1.0 + sy[a] * eta)
        * (1.0 + sz[a] * zeta);
}

double dphi_deta_hex8(int a, double xi, double eta, double zeta) {
    static const int sx[8] = { -1, 1, 1,-1,-1, 1, 1,-1 };
    static const int sy[8] = { -1,-1, 1, 1,-1,-1, 1, 1 };
    static const int sz[8] = { -1,-1,-1,-1, 1, 1, 1, 1 };

    return 0.125 * (1.0 + sx[a] * xi)
        * sy[a]
        * (1.0 + sz[a] * zeta);
}

double dphi_dzeta_hex8(int a, double xi, double eta, double zeta) {
    static const int sx[8] = { -1, 1, 1,-1,-1, 1, 1,-1 };
    static const int sy[8] = { -1,-1, 1, 1,-1,-1, 1, 1 };
    static const int sz[8] = { -1,-1,-1,-1, 1, 1, 1, 1 };

    return 0.125 * (1.0 + sx[a] * xi)
        * (1.0 + sy[a] * eta)
        * sz[a];
}

struct QuadPoint3D {
    double xi, eta, zeta, w;
};

vector<QuadPoint3D> hex8_gauss_2x2x2() {
    vector<QuadPoint3D> q;
    double a = 1.0 / sqrt(3.0);

    for (int ix = 0; ix < 2; ++ix) {
        for (int iy = 0; iy < 2; ++iy) {
            for (int iz = 0; iz < 2; ++iz) {
                q.push_back({
                    ix == 0 ? -a : a,
                    iy == 0 ? -a : a,
                    iz == 0 ? -a : a,
                    1.0
                    });
            }
        }
    }
    return q;
}

#include <set>

void build_profile(SLAE& sys) {
    const Grid3D& g = sys.grid;
    int N = g.nodes_count;
    int NT = 2 * N;

    vector<set<int>> cols(NT);

    for (int kz = 0; kz < g.nz - 1; ++kz) {
        for (int jy = 0; jy < g.ny - 1; ++jy) {
            for (int ix = 0; ix < g.nx - 1; ++ix) {
                int loc[8] = {
                    node_id(ix,     jy,     kz,     g),
                    node_id(ix + 1, jy,     kz,     g),
                    node_id(ix + 1, jy + 1, kz,     g),
                    node_id(ix,     jy + 1, kz,     g),
                    node_id(ix,     jy,     kz + 1, g),
                    node_id(ix + 1, jy,     kz + 1, g),
                    node_id(ix + 1, jy + 1, kz + 1, g),
                    node_id(ix,     jy + 1, kz + 1, g)
                };

                int glob[16];
                for (int a = 0; a < 8; ++a) {
                    glob[a] = id_s(loc[a], N);
                    glob[8 + a] = id_c(loc[a], N);
                }

                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        if (glob[j] < glob[i]) {
                            cols[glob[i]].insert(glob[j]);
                        }
                    }
                }
            }
        }
    }

    sys.A.ig.assign(NT + 1, 0);
    for (int i = 0; i < NT; ++i) {
        sys.A.ig[i + 1] = sys.A.ig[i] + (int)cols[i].size();
    }

    int nz_offdiag = sys.A.ig[NT];
    sys.A.jg.resize(nz_offdiag);
    sys.A.ggl.assign(nz_offdiag, 0.0);
    sys.A.ggu.assign(nz_offdiag, 0.0);
    sys.A.diag.assign(NT, 0.0);

    for (int i = 0; i < NT; ++i) {
        int p = sys.A.ig[i];
        for (int j : cols[i]) {
            sys.A.jg[p++] = j;
        }
    }

    sys.b.assign(NT, 0.0);
    sys.x.assign(NT, 0.0);
    sys.n = NT;
}

int find_pos(const Matrix& A, int i, int j) {
    int left = A.ig[i];
    int right = A.ig[i + 1] - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        int col = A.jg[mid];

        if (col == j) return mid;
        if (col < j) left = mid + 1;
        else right = mid - 1;
    }

    return -1;
}

void add_to_matrix(Matrix& A, int i, int j, double val) {
    if (i == j) {
        A.diag[i] += val;
    }
    else if (j < i) {
        int pos = find_pos(A, i, j);
        if (pos < 0) throw runtime_error("profile mismatch in ggl");
        A.ggl[pos] += val;
    }
    else {
        int pos = find_pos(A, j, i);
        if (pos < 0) throw runtime_error("profile mismatch in ggu");
        A.ggu[pos] += val;
    }
}

void assemble_matrix_rhs(SLAE& sys) {
    Matrix& A = sys.A;
    vector<double>& b = sys.b;

    fill(A.diag.begin(), A.diag.end(), 0.0);
    fill(A.ggl.begin(), A.ggl.end(), 0.0);
    fill(A.ggu.begin(), A.ggu.end(), 0.0);
    fill(b.begin(), b.end(), 0.0);

    const Grid3D& g = sys.grid;
    const HarmonicData& p = sys.phys;

    int N = g.nodes_count;

    double hx = g.hx;
    double hy = g.hy;
    double hz = g.hz;

    double J = hx * hy * hz / 8.0;

    auto quad = hex8_gauss_2x2x2();

    for (int kz = 0; kz < g.nz - 1; ++kz) {
        for (int jy = 0; jy < g.ny - 1; ++jy) {
            for (int ix = 0; ix < g.nx - 1; ++ix) {

                int loc[8] = {
                    node_id(ix,     jy,     kz,     g),
                    node_id(ix + 1, jy,     kz,     g),
                    node_id(ix + 1, jy + 1, kz,     g),
                    node_id(ix,     jy + 1, kz,     g),
                    node_id(ix,     jy,     kz + 1, g),
                    node_id(ix + 1, jy,     kz + 1, g),
                    node_id(ix + 1, jy + 1, kz + 1, g),
                    node_id(ix,     jy + 1, kz + 1, g)
                };

                int gs[8], gc[8];
                for (int a = 0; a < 8; ++a) {
                    gs[a] = id_s(loc[a], N);
                    gc[a] = id_c(loc[a], N);
                }

                double K[8][8] = { {0.0} }; // жесткость
                double M[8][8] = { {0.0} }; // масса
                double bs[8] = { 0.0 };
                double bc[8] = { 0.0 };

                double ex0 = g.x0 + ix * hx;
                double ey0 = g.y0 + jy * hy;
                double ez0 = g.z0 + kz * hz;

                for (const auto& qp : quad) {
                    double xi = qp.xi;
                    double eta = qp.eta;
                    double zeta = qp.zeta;
                    double w = qp.w;

                    double ph[8];
                    double dphix[8];
                    double dphiy[8];
                    double dphiz[8];

                    for (int a = 0; a < 8; ++a) {
                        ph[a] = phi_hex8(a, xi, eta, zeta);

                        dphix[a] = dphi_dxi_hex8(a, xi, eta, zeta) * 2.0 / hx;
                        dphiy[a] = dphi_deta_hex8(a, xi, eta, zeta) * 2.0 / hy;
                        dphiz[a] = dphi_dzeta_hex8(a, xi, eta, zeta) * 2.0 / hz;
                    }

                    double x = ex0 + 0.5 * (xi + 1.0) * hx;
                    double y = ey0 + 0.5 * (eta + 1.0) * hy;
                    double z = ez0 + 0.5 * (zeta + 1.0) * hz;

                    double fs = fs_rhs(x, y, z, p, 1);
                    double fc = fc_rhs(x, y, z, p, 1);

                    for (int i = 0; i < 8; ++i) {
                        for (int j = 0; j < 8; ++j) {
                            K[i][j] += p.lambda *
                                (dphix[i] * dphix[j] +
                                    dphiy[i] * dphiy[j] +
                                    dphiz[i] * dphiz[j]) * J * w;

                            M[i][j] += ph[i] * ph[j] * J * w;
                        }

                        bs[i] += fs * ph[i] * J * w;
                        bc[i] += fc * ph[i] * J * w;
                    }
                }

                // блоки гармонической системы:
                // [ K - w^2 c M + gamma M          -w sigma M ]
                // [ +w sigma M          K - w^2 c M + gamma M ]


                for (int i = 0; i < 8; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        double diag_block = K[i][j]
                            - (p.omega * p.omega) * p.c * M[i][j];

                        double cross_sc = -p.omega * p.sigma * M[i][j];
                        double cross_cs = +p.omega * p.sigma * M[i][j];

                        add_to_matrix(A, gs[i], gs[j], diag_block);
                        add_to_matrix(A, gc[i], gc[j], diag_block);

                        add_to_matrix(A, gs[i], gc[j], cross_sc);
                        add_to_matrix(A, gc[i], gs[j], cross_cs);
                    }

                    b[gs[i]] += bs[i];
                    b[gc[i]] += bc[i];
                }
            }
        }
    }
}

bool is_boundary_node(int i, int j, int k, const Grid3D& g) {
    return i == 0 || j == 0 || k == 0 ||
        i == g.nx - 1 || j == g.ny - 1 || k == g.nz - 1;
}

void apply_dirichlet(Matrix& A, vector<double>& b, int row, double value) {
    int n = (int)A.diag.size();

    for (int i = 0; i < n; ++i) {
        if (i == row) continue;

        if (row < i) {
            int pos = find_pos(A, i, row);
            if (pos >= 0) {
                b[i] -= A.ggl[pos] * value;

                A.ggl[pos] = 0.0;
                A.ggu[pos] = 0.0;
            }
        }
        else {
            int pos = find_pos(A, row, i);
            if (pos >= 0) {
                b[i] -= A.ggu[pos] * value;

                A.ggl[pos] = 0.0;
                A.ggu[pos] = 0.0;
            }
        }
    }

    A.diag[row] = 1.0;
    b[row] = value;
}

void apply_bc_first_kind(SLAE& sys, int testID) {
    const Grid3D& g = sys.grid;
    int N = g.nodes_count;

    for (int k = 0; k < g.nz; ++k) {
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 0; i < g.nx; ++i) {
                if (!is_boundary_node(i, j, k, g)) continue;

                int node = node_id(i, j, k, g);

                double x = g.x0 + i * g.hx;
                double y = g.y0 + j * g.hy;
                double z = g.z0 + k * g.hz;

                apply_dirichlet(sys.A, sys.b, id_s(node, N), us_boundary(x, y, z, testID));
                apply_dirichlet(sys.A, sys.b, id_c(node, N), uc_boundary(x, y, z, testID));
            }
        }
    }
}

void init_grid(SLAE& sys, int nx, int ny, int nz,
    double x0, double x1,
    double y0, double y1,
    double z0, double z1) {
    sys.grid.nx = nx;
    sys.grid.ny = ny;
    sys.grid.nz = nz;

    sys.grid.x0 = x0; sys.grid.x1 = x1;
    sys.grid.y0 = y0; sys.grid.y1 = y1;
    sys.grid.z0 = z0; sys.grid.z1 = z1;

    sys.grid.hx = (x1 - x0) / (nx - 1);
    sys.grid.hy = (y1 - y0) / (ny - 1);
    sys.grid.hz = (z1 - z0) / (nz - 1);

    sys.grid.nodes_count = nx * ny * nz;
    sys.grid.total_unknowns = 2 * sys.grid.nodes_count;
}

void input(SLAE& sys) {
    ifstream file("properties.txt");
    if (!file) throw runtime_error("Cannot open properties.txt");

    file >> sys.grid.nx >> sys.grid.ny >> sys.grid.nz;
    file >> sys.grid.x0 >> sys.grid.x1;
    file >> sys.grid.y0 >> sys.grid.y1;
    file >> sys.grid.z0 >> sys.grid.z1;

    file >> sys.maxiter >> sys.epsilon;

    file >> sys.phys.lambda >> sys.phys.sigma >> sys.phys.c >> sys.phys.omega;

    int method;
    file >> method;

    sys.grid.hx = (sys.grid.x1 - sys.grid.x0) / (sys.grid.nx - 1);
    sys.grid.hy = (sys.grid.y1 - sys.grid.y0) / (sys.grid.ny - 1);
    sys.grid.hz = (sys.grid.z1 - sys.grid.z0) / (sys.grid.nz - 1);

    sys.grid.nodes_count = sys.grid.nx * sys.grid.ny * sys.grid.nz;
    sys.grid.total_unknowns = 2 * sys.grid.nodes_count;

    sys.n = sys.grid.total_unknowns;
}

void print_relative_residual(const SLAE& sys, const string& name) {
    vector<double> r;
    residual(sys.A, sys.x, sys.b, r, sys.n);

    double nb = sqrt(scalar(sys.b, sys.b, sys.n));
    double nr = sqrt(scalar(r, r, sys.n));

    if (nb < 1e-30) {
        cout << name << " relative residual = " << nr << endl;
    }
    else {
        cout << name << " relative residual = " << nr / nb << endl;
    }
}

void check_error(const SLAE& sys, int testId) {
    const Grid3D& g = sys.grid;
    int N = g.nodes_count;

    double max_err_s = 0.0;
    double max_err_c = 0.0;

    for (int k = 0; k < g.nz; ++k) {
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 0; i < g.nx; ++i) {
                int node = node_id(i, j, k, g);

                double x = g.x0 + i * g.hx;
                double y = g.y0 + j * g.hy;
                double z = g.z0 + k * g.hz;

                double us_num = sys.x[id_s(node, N)];
                double uc_num = sys.x[id_c(node, N)];

                double us_ex = us_exact(x, y, z, testId);
                double uc_ex = uc_exact(x, y, z, testId);

                max_err_s = max(max_err_s, fabs(us_num - us_ex));
                max_err_c = max(max_err_c, fabs(uc_num - uc_ex));
            }
        }
    }

    cout << "max error us = " << max_err_s << "\n";
    cout << "max error uc = " << max_err_c << "\n";
}

int main() {
    try {
        int testID = 1;

        SLAE sys;

        input(sys);

        auto build_start = chrono::high_resolution_clock::now();

        build_profile(sys);
        assemble_matrix_rhs(sys);
        apply_bc_first_kind(sys, testID);

        auto build_end = chrono::high_resolution_clock::now();
        auto build_duration = chrono::duration_cast<chrono::milliseconds>(build_end - build_start);

        cout << "Matrix size: " << sys.grid.nx << "x" << sys.grid.ny << "x" << sys.grid.nz << "\n";
        cout << "Matrix build time: " << build_duration.count() << " ms\n";

        cout << "Methods:\n";
        cout << "0 - BiCGSTAB + LU\n";
        cout << "1 - BiCGSTAB\n";
        cout << "2 - LOC_LU\n";
        cout << "3 - LU\n";
        cout << "choose method: ";

        int method = 0;
        cin >> method;

        auto start = chrono::high_resolution_clock::now();

        switch (method) {
        case 0:
            copy_matrix(sys.M, sys.A, sys.n);
            LU(sys.M, sys.n);
            BiCGSTAB_LU(sys);
            break;
        case 1:
            BiCGSTAB(sys);
            break;
        case 2:
            copy_matrix(sys.M, sys.A, sys.n);
            LU(sys.M, sys.n);
            LOC_LU(sys);
            break;
        case 3:
            solveLU_direct(sys);
            print_relative_residual(sys, "Profile LU");
            break;
        default:
            cout << "unknown method\n";
            return 0;
        }

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        cout << "Solution time: " << duration.count() << " ms\n";

        ofstream out("x.txt");
        out << scientific << setprecision(14);
        for (int i = 0; i < sys.n; ++i) {
            out << sys.x[i] << "\n";
        }

        check_error(sys, 1);

        return 0;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}