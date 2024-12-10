
#include <cstdint>

#ifndef
#def IMAGE_HPP_

class image{

    public:
    	image(int width, int height, int nChannels);
        image(const image& other);
        ~image();

        uint8_t* data getChannel(int c);

        int getWidth() const;
        int getHeight() const;
        int getNChannels() const;

    private:
    	int width;
        int height;
        int nChannels;
        uint8_t* data;
}






#endif