#include <iostream>
#include <typeinfo>
#include <math.h>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <ctime>

template<typename T, typename S>
bool equals(const T a, const S b)
{
    return false;
}

template<typename T>
bool equals(const T a, const T b)
{
    return a == b;
}

bool equals(const double a, const double b)
{
    return fabs(a - b) < 0.0001;  
}


template<typename T>
class Vector {
private:
    T* data;
    int size;
public:

    /**
     * @brief Default constructor for a vector.
     * @return An vector with size 0.
     */
    Vector()
    : size(0),
      data(nullptr)
    { }

    /**
     * @brief Create a vector of size n
     * @param n the size of the vector.
     * @return A vector with size n.
     */
    Vector(const int n)
    : data(new T[n]),
      size(n)
    { }


    Vector(const std::initializer_list<T> list)
    : Vector((int)list.size())
    {
        std::uninitialized_copy(list.begin(), list.end(), data);
    }


    /**
     * @brief Copy constructor.
     * @param v
     * @return
     */

    Vector(const Vector<T>& v)
    : Vector(v.size)
    {
        for (int i=0; i<v.size; i++)
            data[i] = v.data[i];
    }

    /**
     * @brief Move constructor.
     * @param v
     * @return
     */
    Vector(Vector&& v)
    : data(v.data),
      size(v.size)
    {
        v.size = 0;
        v.data = nullptr;
    }

    /**
     * @brief Destroy the vector.
     * @return
     */
    ~Vector()
    {
        size=0;
        delete[] data;
    }


    /**
     * @brief Copy assignment.
     * @param other
     */
    Vector<T>& operator=(const Vector<T>& other)
    {
        if (this != &other)
            {
                delete[] data;
                size = other.size;
                data   = new double[other.size];
                for (int i=0; i<other.size; i++)
                    data[i] = other.data[i];
            }
        return *this;
    }

    /**
     * @brief Move assignment.
     * @param other
     */
    Vector<T>& operator=(Vector<T>&& other)
    {
        if (this != &other)
            {
                delete[] data;
                size = other.size;
                data   = other.data;
                other.size = 0;
                other.data   = nullptr;
            }
        return *this;
    }


    /**
     * @brief plus operator.
     * @param other
     */
    Vector<T> operator+(const Vector<T>& other) const
    {
        Vector v(size);
        if(size != other.size)
        {
            throw "Vectors don't have the same size";
        }
        for (auto i=0; i<other.size; i++)
            v.data[i] = data[i] + other.data[i];
    

     //   std::clog << "plus operator" << std::endl;
        return v;
    }
    
    T add_element(int index, const Vector<T>&other) const
    {
        return data[index] + other.data[index];
    }

     /**
     * @brief minus operator.
     * @param other
     */
    Vector<T> operator-(const Vector<T>& other) const
    {
        Vector v(size);
        if(size != other.size)
        {
            throw "Vectors don't have the same size";
        }
        
        for (auto i=0; i<other.size; i++)
             v.data[i] = data[i] - other.data[i];
        

       // std::clog << "minus operator" << std::endl;
        return v;
    }
    
    T subtract_element(int index, const Vector<T>&other) const
    {
        return data[index] - other.data[index];
    }

  /**  Vector operator*(const Vector<T>& scalar) //TODO 
    {
        Vector v(size);
        for (auto i=0; i<size; i++)
            v.data[i] = scalar*data[i];
        return v;
    }**/
    
    Vector<T> operator*(const T scalar) const
    {
        Vector v(size);
        for (auto i=0; i<size; i++)
            v.data[i] = scalar*data[i];
        return v;
    }
    
    T multiply_element(int index, const T scalar) const
    {
        return data[index] *scalar;
    }
    
    T& operator[] (const int index) const
    {
        if (index >= size || index < 0)
            throw "Index out of bounds";

        return data[index];
    }
    
    int get_size() const 
    { 
        return size; 
    }
    

    void print() const
    {
        std::cout << "[";
        for(int x = 0; x < size; x++) {
            if (x != 0)
                std::cout << ", ";
            std::cout << data[x];
        }
        std::cout << "]" << std::endl;        
    }   
};

template <typename T>
bool equals(const Vector<T>& a, const Vector<T>& b)
{
    if (a.get_size() != b.get_size())
        return false;
    for(int i = 0; i < a.get_size(); i++)
    {
        if(!equals(a[i], b[i]))
            return false;
    }
       
    return true;
}
    
template <typename T>
Vector<T> operator*(const T scalar, const Vector<T> &v) {
    return v * scalar;
}



/**
* @brief Function that returns the dot product of two vectors.
* @param two vectors of the same length.
* @return the dot product.
*/

template<typename T>
T dot(const Vector<T>& l, const Vector<T>& r)
{
    if(l.get_size() != r.get_size())
    {
        throw "Vectors don't have the same size";
    }
        
    // Calculate the dot product
    double d=0;
    for (auto i=0; i<l.get_size(); i++)
        d += l[i]*r[i];
    return d;
}

template<typename T>
class Matrix {
private:
    int size_m;
    int size_n;
    T** data;
public:
    Matrix(const int m, const int n)
    : size_n(n),
      size_m(m),
      data(new T*[m])
    {
        for(int i = 0; i < m; i++)
            data[i] = new T[n]{0};
    }
     
    
    ~Matrix()
    {
        size_m = 0;
        size_n = 0;
        for(int i = 0; i < size_m; i++)
            delete[] data[i];
        delete[] data;
    }
    
    /**
     * @brief Copy constructor.
     * @param v
     * @return
     */
    Matrix(const Matrix<T>& m)
    : Matrix<T>(m.size_m, m.size_n)
    {
        for (int i=0; i<m.size_m; i++)
            for (int j=0; i<m.size_n; j++)
                data[i][j] = m.data[i][j];
    }

    /**
     * @brief Move constructor.
     * @param v
     * @return
     */
    Matrix(Matrix<T>&& m)
    : data(m.data),
      size_m(m.size_m),
      size_n(m.size_n)
    {
        m.size_m = 0;
        m.size_n = 0;
        m.data = nullptr;
    }
    
    /**
     * @brief Copy assignment.
     * @param other
     */
    Matrix<T>& operator=(const Matrix<T>& other)
    {
        if (this != &other)
        {
            //TODO check size
            for (int i=0; i<other.size_m; i++)
                for (int j=0; i<other.size_n; j++)
                    data[i][j] = other.data[i][j];
        }
        return *this;
    }

    /**
     * @brief Move assignment.
     * @param other
     */
    Matrix<T>& operator=(Matrix<T>&& other)
    {
        if (this != &other)
            {
                delete[] data;
                //TODO check size
                for (int i=0; i<other.size_m; i++)
                    for (int j=0; i<other.size_n; j++)
                        data[i][j] = other.data[i][j];
                data = other.data;
                
                other.size_m = 0;
                other.size_n = 0;
                other.data   = nullptr;
            }
        return *this;
    }
    
    T& operator[] (const std::initializer_list<int> index) 
    {
        if (index.size() != 2)
            throw "Needs 2 elements for the index";
            
        int m = index.begin()[0];
        int n = index.begin()[1];
        if (m < 0 || n < 0 || m >= size_m || n >= size_n)
            throw "Index out of bounds";
            
        return data[m][n];
    }
    
    Vector<T> matvec(const Vector<T> &v) const
    {
        Vector<T> result(size_m);
        if (v.get_size() != size_n)
            throw "Wrong size";
            
        for (int i=0; i< size_m; i++) {
            result[i] = 0;
            for (int j=0; j< size_n; j++) {
                result[i] += data[i][j] * v[j];
            }
        }
        
        return result;
    }
    
    T matvec_element(int index, const Vector<T> & v) const
    {
        T result = 0;
        for (int j=0; j< size_n; j++) {
            result += data[index][j] * v[j];
        }
    }
    

    Vector<T> operator*(const Vector<T> &v) const
    {
        return matvec(v);
    }
    
    void print() const
    {
        for(int y = 0; y < size_m; y++) {
            std::cout << "[";
            for(int x = 0; x < size_n; x++) {
                if (x != 0)
                    std::cout << ", ";
                std::cout << data[y][x];
            }
            std::cout << "]" << std::endl;
        }
        
    }
    
    int get_size(int dim) const 
    { 
        if (dim == 0) {
            return size_m;
        } else if (dim == 1) {
            return size_n;
        } else {
            throw "Dimension must be 0 or 1";
        }
    }
 
};

template <typename T>
bool equals(const Matrix<T>& a, const Matrix<T>& b)
{
    if (a.get_size(0)!= b.get_size(0) || a.get_size(1) != b.get_size(1))
        return false;
    for(int i = 0; i < a.get_size(0); i++)
    {
        for(int j = 0; j < a.get_size(1); j++)
        {
            if(!equals(a[i][j], b[i][j]))
                return false;
        }
    }
       
    return true;
}


template<typename T>
int cg2(
    const Matrix<T> &A, const Vector<T> &b, Vector<T> &x, T tol, int maxiter)
{
    Vector<T> r = b - A * x;
    Vector<T> p = r; 
    int size = x.get_size();
    //Vector<T> x_n(size);
    //Vector<T> r_n(size); 
    Vector<T> temp(size); 
    for(int k = 0; k < maxiter; k++)
    {
       // std::cout << "cg" << k << " / " << maxiter << std::endl;
        for (int i = 0; i < size; i++)
        {
            temp[i] = A.matvec_element(i, p);
        }
        
        T dot_r_r = dot(r, r);
        T alpha =  dot_r_r / dot(temp, p);
        
        for (int i = 0; i < size; i++)
        {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * temp[i];
        }

        T dot_rn_rn = dot(r, r);
        if (dot_rn_rn < tol*tol) {
            return k;
            
        }
        T beta = dot_rn_rn / dot_r_r;
        p = r + beta * p;
        for (int i = 0; i < size; i++)
        {
            p[i] = r[i] + beta * p[i];
        }
    }
    return -1;
    
}




template<typename T>
int cg(
    const Matrix<T> &A, const Vector<T> &b, Vector<T> &x, T tol, int maxiter)
{
    Vector<T> r = b - A * x;
    Vector<T> p = r;

    for(int k = 0; k < maxiter; k++)
    {
       // std::cout << "cg" << k << " / " << maxiter << std::endl;
        T alpha = dot(r, r) / dot(A * p, p);
        Vector<T> x_n = x + alpha * p;
        Vector<T> r_n = r - alpha * (A * p);
        if (dot(r_n, r_n) < tol*tol) {
            x = x_n;
            return k;
            
        }
        T beta = dot(r_n, r_n) / dot(r, r);
        p = r_n + beta * p;
        r = r_n;
        x = x_n;
    }
    return -1;
}

class Heat1D
{
private:
    Matrix<double> M;
    Vector<double> u_0;
    int points;
    double alpha;
    double dt;
    double dx;
    
public:
    Heat1D(double p_alpha, int m, double p_dt) :
        M(m,m),
        points(m),
        alpha(p_alpha),
        dt(p_dt),
        dx( 1.0/(m+1)),
        u_0(m)
    {
        for (int i = 0; i < m; i++)
        {
            int l = i - 1;
            int r = i + 1;
            double s = alpha * dt / (dx * dx);
            M[{i,i}] = 1 + 2 * s;
            if (l >= 0)
                M[{i,l}] = -1 * s;
            if (r < m)
                M[{i,r}] = -1 * s;
        }

        for (int i = 0; i < m; i++)
        {        
            u_0[i] = sin(M_PI * (i + 1) * dx); //+1 because cell centered
        }
    }
    
    
    Vector<double> exact(double t) const
    {
        Vector<double> result(points);
        for (int i = 0; i < points; i++)
        {
            result[i] = exp(-M_PI * M_PI * alpha * t) * u_0[i];
        }

        return result;
    }
    
    Vector<double> solve(double t_end) const
    {
        Vector<double> x = u_0;
        int steps = t_end / dt - 1;
        
        for (int i = 0; i < steps; i++)
        {
            Vector<double> b = x;    
            if (cg<double>(M, b, x, 0.0001, 100) < 0) 
                throw "Error";
        }
        
        return x;
    }
    
        Vector<double> solve2(double t_end) const
    {
        Vector<double> x = u_0;
        int steps = t_end / dt - 1;
        
        for (int i = 0; i < steps; i++)
        {
            Vector<double> b = x;    
            if (cg2<double>(M, b, x, 0.0001, 100) < 0) 
                throw "Error";
        }
        
        return x;
    }
};


class Heat2D
{
private:
    Matrix<double> M;
    Vector<double> u_0;
    int points;
    double alpha;
    double dt;
    double dx;
    
public:
    Heat2D(double p_alpha, int m, double p_dt) :
        M(m * m,m * m),
        points(m * m),
        alpha(p_alpha),
        dt(p_dt),
        dx( 1.0/(m+1)),
        u_0(m * m)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                int index = i * m + j;
                int d = i - 1;
                int u = i + 1;
                int l = j - 1;
                int r = j + 1;
                double s = alpha * dt / (dx * dx);
                M[{index,index}] = 1 + 2 * 2 * s;
                if (l >= 0)
                    M[{index,index - 1}] = -1 * s;
                if (r < m)
                    M[{index,index + 1}] = -1 * s;
                if (d >= 0)
                    M[{index,index - m}] = -1 * s;
                if (u < m)
                    M[{index,index + m}] = -1 * s;
            }
        }

        for (int i = 0; i < m; i++)
        {  
            double u_i = sin(M_PI * (i + 1) * dx);
            for (int j = 0; j < m; j++)
            {      
                u_0[i * m + j] = u_i + sin(M_PI * (j + 1) * dx); //+1 because cell centered
            }
        }
        u_0.print();
    }
    
    
    Vector<double> exact(double t) const
    {
        Vector<double> result(points);
        for (int i = 0; i < points; i++)
        {
            result[i] = exp(-2*M_PI * M_PI * alpha * t) * u_0[i];
        }

        return result;
    }
    
    Vector<double> solve(double t_end) const
    {
        Vector<double> x = u_0;
        int steps = t_end / dt - 1;
        
        for (int i = 0; i < steps; i++)
        {
            std::cout << i << " / " << steps << std::endl;
            Vector<double> b = x;
  clock_t begin = clock();    
            if (cg2<double>(M, b, x, 0.001, 100) < 0) 
                throw "Error";
                  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  std::cout << elapsed_secs << std::endl;
        }
        
        return x;
    }
};




void test_vector_constructor(void)
{
    Vector<double> A;
    Vector<double> B(0);
    Vector<double> C({});
    Vector<int> E;
    Vector<double> F(1);
    Vector<double> G({1});
    Vector<double> H({1,2,3});
    Vector<double> I({1,2,3});
    Vector<double> J({1,2,3.1});
    Vector<double> K({0.9,2,3});
    Vector<double> L = I;
    assert(A.get_size() == 0);
    assert(B.get_size() == 0);
    assert(C.get_size() == 0);
    assert(F.get_size() == 1);
    assert(H.get_size() == 3);
    assert(equals(A,B));
    assert(equals(A,C));
    assert(!equals(A,E));
    
    assert(!equals(A,F));
    assert(!equals(A,G));
    assert(!equals(F,G));
    assert(!equals(G,H));
    assert(equals(H,I));
    assert(!equals(I,J));
    assert(!equals(I,K));
    assert(equals(L,I));
    assert(&L != &I);
    I[0] = 0.9;
    assert(!equals(L,I));
    assert(equals(I,K));
}

void test_vector_operations(void)
{
    Vector<double> A({1,2,3});
    Vector<double> B({2,4,-1});
    
    Vector<double> expected({3,6,2});
    assert(equals(expected, A+B));
    
    expected = Vector<double>({-1,-2,4});
    assert(equals(expected, A-B));
    
    expected = Vector<double>({4, 8, -2});
    assert(equals(expected, 2.0 * B));
    assert(equals(expected, B * 2.0));

    assert(equals(7.0, dot(A,B)));
    
    bool exception = false;
    Vector<double> C({2,4});
    try {
        A+C;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;
    
    try {
        C+A;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;
    
    try {
        A-C;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;
    
    try {
        C-A;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;
    
    try {
        dot(A,C);
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;
    
    try {
        dot(C,A);
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);    
}

void test_matrix(void)
{
    Matrix<double> test(3,4);
    assert(equals(3, test.get_size(0)));
    assert(equals(4, test.get_size(1)));
    assert(equals(0.0, test[{1,2}]));
    assert(equals(0.0, test[{0,0}]));
    test[{0,0}] = -1.1;
    test[{2,3}] = 1.0;
    assert(equals(-1.1, test[{0,0}]));
    assert(equals(1.0, test[{2,3}]));
    
    bool exception = false;
    try {
        test[{3,0}] = 1.0;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;

    try {
        test[{0,4}] = 1.0;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
}

void test_matrix_vector(void)
{
    Matrix<double> M(3,4);
    Vector<double> v({1,2,3,4});
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            M[{i,j}] = i + j;
        }
    }
    
    Vector<double> expected({20, 30, 40});
        
        
    Vector<double> result = M*v;
    result.print();

    assert(equals(expected, M*v));
    
    Vector<double> v2({1,2,3});
    bool exception = false;
    try {
        M*v2;
    } catch (char const* e) {
        exception = true;
    }
        assert(exception);
        
    exception = false;
    Vector<double> v3({1,2,1,1,3});
    try {
        M*v3;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    
}

/**
 * @brief Test the code.
 */
void test(void)
{
    test_vector_constructor();
    test_vector_operations();
    test_matrix();
    test_matrix_vector();
    std::cout << "All test passed!" << std::endl;
}

template<typename T>
T error(Vector<T> a, Vector<T> b)
{
    T result;
    for (int i = 0; i < a.get_size(); i++)
    {
        result += fabs(a[i] - b[i]);
    }
    return result;
    
}

int main(){
  //  test();
    Heat1D test(0.3125, 10, 0.0001);
    
    try {
    Vector<double> a = test.exact(01);
    Vector<double> b = test.solve(1);
    Vector<double> c = test.solve2(1);
    a.print();
    std::cout << " " << std::endl;
    b.print();
        std::cout << " " << std::endl;
    c.print();
    std::cout << "Error head1D: " << error(a,b) << std::endl;
    std::cout << "Error head1D2: " << error(c,b) << std::endl;
  //  return 0;
    Heat2D test2(0.3125, 99, 0.001);
    Vector<double> a2 = test2.exact(0.5);
    std::cout << " "  << std::endl;
    a2.print();
    std::cout << " "  << std::endl;
    std::cout << "start solving:"  << std::endl;
    Vector<double> b2 = test2.solve(0.5);
        std::cout << " "  << std::endl;
            b2.print();
        std::cout << "Error head2D: " << error(a2,b2) << std::endl;
        
            Vector<double> dif = a2 - b2;
            for (int i = 0; i < 10; i ++)
            {
                std::cout  << std::endl;
                for (int j = 0; j < 10; j ++)
            {
                std::cout << dif[i*10 + j] << ", ";
                
            }
                
            }
    } catch(const char * e) {
            std::cout << e << std::endl;
    }
    

    return 0;
}
