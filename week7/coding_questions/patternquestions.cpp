//Q1 Create lower triangular
#include<iostream>
using namespace std;

int  main(){
cout<<"enter the no"<<endl;
int n,res;
n=4;
//lower triangular
  for(int i=0;i<n;i++){
    for(int j=0;j<=i;j++){
      cout<<"1";

    }
    cout<<endl;
  }
cout<<endl;

cout<<endl;
  //upper triangular
for(int i=0;i<n;i++){
    for(int j=n-1;j>=i;j--){
      cout<<"1";

    }
    cout<<endl;
  }


  //pyramid
for (int i = 1; i <= n; ++i) {
        // Print spaces
        for (int space = 1; space <= n - i; ++space) {
            cout << " ";
        }

        // Print stars
        for (int star = 1; star <= 2 * i - 1; ++star) {
            cout << "*";
        }

        cout << endl;
    }
  return 0;
}