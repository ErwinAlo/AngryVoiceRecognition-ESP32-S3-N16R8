#include <MD_MAX72xx.h>  // Incluye la biblioteca para controlar la matriz de LEDs MAX72XX
#include "emotions.h" // Incluye el archivo de encabezado para las funciones relacionadas con las emociones

//Declaraciones
//*********************************************************************************

//Mapeado de emociones
static uint8_t set_map[4][8] = 
{
  { // Other 
    B00000000,
    B01111110,
    B01000010,
    B01000010,
    B01000010,
    B01000010,
    B01111110,
    B00000000
  },
  { // Angry 
    B00000000,
    B00110010,
    B00010100,
    B00000100,
    B00000100,
    B00010100,
    B00110010,
    B00000000
  },
  { //No Sound
    B10000001,
    B00000000,
    B00000000,
    B00000000,
    B00000000,
    B00000000,
    B00000000,
    B10000001
  }
  // { // Happy
  //   B00000000,
  //   B00110100,
  //   B00110010,
  //   B00000010,
  //   B00000010,
  //   B00110010,
  //   B00110100,
  //   B00000000
  // },
  // { // Neutral
  //   B00000000,
  //   B00110000,
  //   B00110010,
  //   B00000010,
  //   B00000010,
  //   B00110010,
  //   B00110000,
  //   B00000000
  // },
  // { // Sad
  //   B00000000,
  //   B00110010,
  //   B00110100,
  //   B00000100,
  //   B00000100,
  //   B00110100,
  //   B00110010,
  //   B00000000
  // },
  // { // Surprise
  //   B00000000,
  //   B01100000,
  //   B01101111,
  //   B00001001,
  //   B00001001,
  //   B01101111,
  //   B01100000,
  //   B00000000
  // }
};

//Funciones
//*********************************************************************************

//Función para dibujar las emociones en matriz
//*********************************************************************************
void emotion(int sets, int colMat[1], uint8_t player, MD_MAX72XX &mxObj)
{  
  //Inicializa contador
  int cont = 0;

  //For para recorrer las columnas de las primeras matrices
  for(int i=0; i<8; i++)
  {
    mxObj.setRow(colMat[player], i, set_map[sets][cont]);
    cont = cont+1;
  }
}

//Función para dibujar puntos en ausencia de sonido
//*********************************************************************************
void noSound(int colMat[1], uint8_t player, MD_MAX72XX &mxObj)
{  
  //Inicializa contador
  int cont = 0;

  //For para recorrer las columnas de las primeras matrices
  for(int i=0; i<8; i++)
  {
    mxObj.setRow(colMat[player], i, set_map[2][cont]);
    cont = cont+1;
  }
}