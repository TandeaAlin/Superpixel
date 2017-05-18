# Superpixel

Structura Slic de la inceputul fisierului contine:
  - dimensiunea unui grid facuta in functie de dimensiunea imaginii si a numarului de pixeli; astfel, se incearca distribuirea 
    superpixelilor la distante echidistante; gridul are dimensiunea step x step, unde step este calculat cu formula din documnetul
    proiectului
  - constantM reprezinta variabila m intalnita in formula de calculare a distantei
  - labels este variabila in care numarul superpixelului(clusterului) la care este asociat fiecare pixel din imagine
  - distances este variabila in care se retine cea mai mica distanta gasita pana in prezent fata de un superpixel al fiecarui pixel din 
    agine
  - occurrencesCluster retine numarul de pixeli care sunt asociati fiecarui cluser(superpixel)
  - clusters este variabila ce retine informatiile despre fiecare superpixel din algoritm; retine informatii despre culoarea acestuia si 
  pozitia sa in imagine
  
 Metoda findMinumum(Mat img, Point center) gaseste pixelul cu cel mai mic gradient intr-o vecnatate de 3X3. Gradientul se refera la 
 modificarile de intensitate avute intre pixelul curent fata de pixelii situati deasupra si dedesuptul lui in imaginea initiala.
 
 Metoda initialize(Mat src, int superpixelNr, Slic* slic, int constantM) determina initializarea structurii Slic tinand cont de numarul de
 pizeli si de constanta M. Initilizarea pune toate valorile din labels pe -1, din distance pe valoarea maxima FLT_MAX, din occurrencesCluster
 pe valoarea 0; vectorul clusters este initializat cu informatii ale pixelilor echidistanti in functie de grandientul minim al pixelilor
 din vecinatate.
 
 Metoda clear(Slic* slic) reseteaza toate campurile structurii.
 
 Metoda findDistance(Point position, Vec3d color, int cluster, Slic* slic) calculeaza distanta dintre un pixel reprezentat prin culoare(color)
 si pozitie in imaine(position) fata de un anumit cluster identificat prin indexul sau in vectorul asociat in structura. Formula este gasita
 in documentul proiectului.
 
 Metoda displayClusters(Mat src, Slic* slic) am folosit-o doar pentru a ma asigura ca pozitia initiala a superpixelilor este distribuita la 
 distante egale; nu are niciun rol in algoritm.
 
 Metoda generateSuperpixel(Mat src, int superpixelNr, int constantM, Slic* slic) ar reprezenta metoda ce contine algoritmul. Momentan, 
 contine doar partea de initializare.
 
 Metoda colorByClusters((Mat src, Slic* slic) contine metoda ce trebuie folosita la sfarsitul algoritmului, recolorand pixelii in functie
 de superpixelii carora au fost asociati. Colorarea presupune atribuirea pixelilor asociati unui superpixel a culorii determinate prin
 medierea culorilor gasite in acesti pixeli.
 
 Metodele findLABSpaceValue(double t, double eps), BGR2LAB(Vec3b pixel),RGB2LABConversion(Mat src) le-am folosit pentru a realiza manual
 convrsia din spatiul de culoare BGR in spatiul de culoare LAB. Rezultatul nu este cel obtinut cu cel al mtodei predefinite din 
 opencv. Modul de calculare poate fi gasit pe site-ul lor de conversii. As pune link-ul dar momentan nu imi permite netul din camin
 sa intru pe site-ul lor.
 
 
