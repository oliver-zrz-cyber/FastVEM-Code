#include "dfn.h"
#include "func.h"
#include "mac.h"


void render_loop(RGB_Screen &screen);

void add_external_force(MAC_Grid& mac);

void velocity_advection(MAC_Grid& mac);

void grid_pressure_projection(MAC_Grid& mac);

void simple_CFL(MAC_Grid& mac);

void density_advection(MAC_Grid& mac);

void init_flip(MAC_Grid& mac);

void ptogrid(MAC_Grid& mac);

void gridtop(MAC_Grid& mac,Real flip_alpha);

void flip_advection(MAC_Grid& mac);

void render_loop_vem(RGB_Screen &screen,std::string domain);







